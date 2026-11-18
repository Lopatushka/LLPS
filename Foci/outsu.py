import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
import argparse
import shutil
import tomllib
from skimage import io, filters, morphology, measure, segmentation
from skimage.feature import peak_local_max
from skimage.restoration import rolling_ball
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# Histogram plots: number of bins and upper intensity percentile shown
HIST_BINS = 200
HIST_PERCENTILE = 99.99


# Width (px) of the ring around the foci that is excluded from the nucleoplasm:
# microscope blur spreads foci light into it and would inflate the nucleoplasm mean
NUCLEOPLASM_RING_PX = 2


def signal_metrics(original, nucleus, labels):
    """Partition coefficient per focus and fraction of nuclear signal in foci.

    Measured on the raw image: rolling-ball subtraction would remove the
    nucleoplasm signal itself. Nucleoplasm = nucleus minus the foci and a
    NUCLEOPLASM_RING_PX ring around them.
    """
    foci = labels > 0
    near_foci = ndi.binary_dilation(foci, structure=np.ones((3, 3)),
                                    iterations=NUCLEOPLASM_RING_PX) if foci.any() else foci
    nucleoplasm = nucleus & ~near_foci

    nucleus_mean = float(original[nucleus].mean())
    nucleoplasm_mean = float(original[nucleoplasm].mean()) if nucleoplasm.any() else np.nan

    total = float(original[nucleus].sum())
    fraction_in_foci = float(original[foci].sum()) / total if total > 0 else np.nan

    n = int(labels.max())
    if n and nucleoplasm_mean > 0:
        raw_means = ndi.mean(original, labels=labels, index=np.arange(1, n + 1))
        partition = np.asarray(raw_means) / nucleoplasm_mean
    else:
        partition = np.full(n, np.nan)

    return dict(partition_coefficient=partition,  # index i = ROI label i + 1
                nucleus_mean_intensity=nucleus_mean,
                nucleoplasm_mean_intensity=nucleoplasm_mean,
                fraction_signal_in_foci=fraction_in_foci)


def gini(values):
    """Gini coefficient of the focus areas of one cell, rescaled by n/(n-1).

    0 = all foci the same size, 1 = one focus holds all the area. The n/(n-1)
    factor makes the maximum 1 for any number of foci, so cells with different
    foci counts are comparable. NaN for fewer than 2 foci.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n < 2 or x.sum() == 0:
        return np.nan
    g = 2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n
    return float(g * n / (n - 1))


def spacing_metrics(rois, nucleus):
    """Nearest-neighbour distance per focus and Clark-Evans ratio per cell.

    nn_distance: centroid-to-centroid distance (px) from each focus to the
    closest other focus of the same nucleus. Clark-Evans ratio R = observed
    mean nn_distance / mean expected for the same number of randomly placed
    points in the nucleus: R < 1 clustered, ~1 random, > 1 evenly spaced.
    The expected value uses Donnelly's edge correction
    0.5*sqrt(A/n) + (0.0514 + 0.041/sqrt(n)) * P/n, because foci near the
    nuclear edge have no neighbours outside it (A, P = nucleus area and
    perimeter). NaN for fewer than 2 foci.
    """
    n = len(rois)
    if n < 2:
        return np.full(n, np.nan), dict(mean_nn_distance=np.nan, clark_evans_ratio=np.nan)
    points = rois[["centroid-0", "centroid-1"]].to_numpy(float)
    nn = cKDTree(points).query(points, k=2)[0][:, 1]  # k=1 is the point itself
    area = float(nucleus.sum())
    perim = float(measure.perimeter(nucleus))
    expected = 0.5 * np.sqrt(area / n) + (0.0514 + 0.041 / np.sqrt(n)) * perim / n
    mean_nn = float(nn.mean())
    return nn, dict(mean_nn_distance=mean_nn, clark_evans_ratio=mean_nn / expected)

def process_image(image_path,
                  threshold_factor=1,
                  sigma=0.6,
                  min_area=2,
                  min_distance=3,
                  rolling_ball_radius=0,
                  segmentation_method="watershed"):
    """Segment the foci of one image; return the intermediate images and the ROI table."""

    image_path = Path(image_path)

    # 1. Load and convert to float
    original = io.imread(image_path)
    if original.ndim != 2:
        raise ValueError(f"{image_path.name}: expected 2D grayscale TIFF, got {original.shape}")
    original = original.astype(np.float32)
    if not np.isfinite(original).all():
        raise ValueError(f"{image_path.name}: image contains NaN or infinity")

    # Nucleus = non-zero pixels (everything outside the nucleus is set to 0)
    nucleus = original > 0
    if not nucleus.any():
        raise ValueError(f"{image_path.name}: image has no non-zero (nucleus) pixels")

    # 2. Background subtraction (rolling ball); 0 disables it
    if rolling_ball_radius > 0:
        background = rolling_ball(original, radius=rolling_ball_radius)
        corrected = np.clip(original - background, 0, None)
    else:
        background = np.zeros_like(original)
        corrected = original.copy()

    # 3. Smoothing
    smoothed = filters.gaussian(
        corrected,
        sigma=sigma,
        preserve_range=True
    )

    # 4. Otsu threshold, computed on the nucleus pixels only (the zeros
    #    outside the nucleus would otherwise pull the threshold down)
    otsu_threshold = float(filters.threshold_otsu(smoothed[nucleus]))
    threshold = otsu_threshold * threshold_factor

    binary_mask = (smoothed > threshold) & nucleus

    mask = morphology.remove_small_objects(
        binary_mask,
        min_size=min_area,
        connectivity=2  # 8-connectivity, same as the ROIs
    )

    # 5. Split the mask into foci: watershed on the distance transform
    #    (splits touching foci) or plain connected regions
    distance = ndi.distance_transform_edt(mask)

    if segmentation_method == "watershed":
        coordinates = peak_local_max(
            distance,
            min_distance=min_distance,
            labels=mask
        )

        markers = np.zeros_like(mask, dtype=np.int32)

        for i, (y, x) in enumerate(coordinates, start=1):
            markers[y, x] = i

        labels = segmentation.watershed(
            -distance,
            markers,
            mask=mask,
            connectivity=2  # also reach diagonally connected pixels
        )

        # Merge fragments smaller than min_area into the touching ROI: the
        # remaining ROIs are grown back over the mask by a second watershed.
        # Fragments that touch no larger ROI stay 0 and are removed below.
        big = morphology.remove_small_objects(labels, min_size=min_area)
        labels = segmentation.watershed(-distance, big, mask=mask, connectivity=2)

        # A small focus split only into fragments < min_area has no ROI left
        # to merge into: keep it as one unsplit ROI (its connected region)
        unsplit = mask & (labels == 0)
        if unsplit.any():
            regions = measure.label(unsplit, connectivity=2)
            labels[unsplit] = regions[unsplit] + labels.max()
    else:
        labels = measure.label(mask, connectivity=2)

    # Drop ROIs smaller than min_area (isolated small objects),
    # renumber 1..N and keep the mask consistent with the ROIs
    labels = morphology.remove_small_objects(labels, min_size=min_area)
    labels = segmentation.relabel_sequential(labels)[0]
    mask = labels > 0

    # 6. Measurements
    properties = measure.regionprops_table(
        labels,
        intensity_image=corrected,  # background-subtracted if rolling_ball_radius > 0
        properties=[
            "label",
            "area",
            "centroid",
            "equivalent_diameter_area",
            "eccentricity",
            "solidity",
            "perimeter",
            "mean_intensity",
            "max_intensity",
            "min_intensity"
        ]
    )

    rois = pd.DataFrame(properties)
    rois.insert(0, "image_id", image_path.stem)

    # 7. Partition coefficient and fraction of signal in foci (raw image)
    metrics = signal_metrics(original, nucleus, labels)
    rois["partition_coefficient"] = metrics["partition_coefficient"][rois["label"].to_numpy(int) - 1]

    # Nearest-neighbour distances and Clark-Evans ratio
    nn, spacing = spacing_metrics(rois, nucleus)
    rois["nn_distance"] = nn
    metrics.update(spacing)

    return dict(metrics=metrics, original=original, nucleus=nucleus, background=background,
                corrected=corrected, smoothed=smoothed, mask=mask,
                distance=distance, labels=labels,
                otsu_threshold=otsu_threshold, threshold=threshold, rois=rois)


def _draw_histogram_ax(ax, result, threshold_factor, log=False):
    values = result["smoothed"].ravel()
    lo = float(values.min())
    hi = float(np.percentile(values, HIST_PERCENTILE))
    if hi <= lo:
        hi = float(values.max())
    if hi <= lo:
        hi = lo + 1

    ax.hist(values, bins=np.linspace(lo, hi, HIST_BINS + 1), density=True,
            color="gray", alpha=0.5, label="Pixel intensities")
    ax.axvline(result["otsu_threshold"], color="tab:blue", linestyle="--", lw=1.5,
               label=f"Otsu threshold = {result['otsu_threshold']:.2f}")
    ax.axvline(result["threshold"], color="tab:red", lw=2,
               label=f"Otsu × {threshold_factor} = {result['threshold']:.2f} [USED]")

    if log:
        ax.set_yscale("log")

    ax.set_title("Histogram + threshold (" + ("log Y" if log else "linear Y") + ")")
    ax.set_xlabel("Smoothed, background-corrected intensity")
    ax.set_ylabel("Probability density")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)


def draw_histogram(result, name, threshold_factor, save_path, figure_dpi):
    """Log-scale intensity histogram with the Otsu threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_histogram_ax(ax, result, threshold_factor, log=True)
    fig.suptitle(f"{name}: intensity distributions", fontsize=14)
    fig.tight_layout()
    try:
        fig.savefig(save_path, dpi=figure_dpi, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_results(result, name, config, save_path):
    """Six-panel overview figure of the Otsu segmentation."""
    original = result["original"]
    labels = result["labels"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    panels = [
        (original, "gray", "Original image"),
        (result["background"], "gray",
         f"Rolling-ball background (r={config['rolling_ball_radius']})"),
        (result["corrected"], "gray", "Background corrected (before smoothing)"),
        (result["mask"], "gray", "Binary mask"),
        (result["distance"], "magma", "Distance transform"),
    ]

    for ax, (img, cmap, title) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    # Sixth panel: ROI boundaries on the background-corrected image
    ax = axes[1, 2]
    corrected = result["corrected"].astype(float)
    span = corrected.max() - corrected.min()
    corrected = (corrected - corrected.min()) / (span if span > 0 else 1)
    ax.imshow(segmentation.mark_boundaries(corrected, labels))
    ax.set_title(f"Segmented ROIs ({config['segmentation']}, N={len(result['rois'])})")
    ax.axis("off")

    fig.suptitle(
        f"{name} | Otsu × {config['threshold_factor']} | σ={config['sigma']}",
        fontsize=15
    )
    fig.tight_layout()
    try:
        fig.savefig(save_path, dpi=config["figure_dpi"], bbox_inches="tight")
    finally:
        plt.close(fig)


DEFAULT_CONFIG = Path(__file__).with_name("config_outsu.toml")

# How the foci mask is split into ROIs
SEGMENTATION_METHODS = ("watershed", "connected")

CONFIG_TYPES = {
    "sigma": (int, float),
    "threshold_factor": (int, float),
    "min_area": int,
    "min_distance": int,
    "segmentation": str,
    "rolling_ball_radius": (int, float),
    "save_figures": bool,
    "figure_dpi": int,
}


def load_config(path):
    """Read the TOML config and check that it has exactly the expected keys."""
    with open(path, "rb") as f:
        config = tomllib.load(f)

    missing = CONFIG_TYPES.keys() - config.keys()
    unknown = config.keys() - CONFIG_TYPES.keys()
    if missing or unknown:
        raise ValueError(
            f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}"
        )

    for key, types in CONFIG_TYPES.items():
        value = config[key]
        # bool is a subclass of int: reject true/false for numeric keys
        if not isinstance(value, types) or (types is not bool and isinstance(value, bool)):
            raise TypeError(f"{path}: '{key}' has invalid value {value!r}")

    if config["sigma"] < 0 or config["threshold_factor"] <= 0:
        raise ValueError(f"{path}: sigma must be >= 0 and threshold_factor > 0")
    if config["min_area"] < 1 or config["min_distance"] < 1 or config["figure_dpi"] < 1:
        raise ValueError(f"{path}: min_area, min_distance and figure_dpi must be >= 1")
    if config["segmentation"] not in SEGMENTATION_METHODS:
        raise ValueError(f"{path}: segmentation must be one of {SEGMENTATION_METHODS}")
    if config["rolling_ball_radius"] < 0:
        raise ValueError(f"{path}: rolling_ball_radius must be >= 0 (0 = no background subtraction)")

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Otsu threshold foci segmentation of single-nucleus TIFF images.

For every image: optional rolling-ball background subtraction
(rolling_ball_radius, 0 = off) -> Gaussian smoothing (sigma) -> mask = pixels above
Otsu threshold (computed on the nucleus = non-zero pixels) * threshold_factor -> removal of objects smaller than
min_area -> foci ROIs (segmentation = "watershed": distance transform +
watershed to split touching foci; "connected": connected regions) ->
shape and intensity measurements of every focus.""",
        epilog="""\
input:
  A folder with one 2D grayscale image per nucleus (*.tif / *.tiff),
  e.g. the single-nucleus images saved by FIJI/nuclei_manual_segment.py.

output (subfolders of the --output folder, same layout as gmm.py):
  roi_measurements/<image>_ROIs.csv   one row per focus: image_id, label,
                                      area, centroid-0/-1,
                                      equivalent_diameter_area, eccentricity,
                                      solidity, perimeter,
                                      mean/max/min_intensity,
                                      partition_coefficient, nn_distance
                                      (to the nearest focus). Sizes are in
                                      pixels. Intensities are measured on the
                                      background-subtracted image if
                                      rolling_ball_radius > 0, else raw;
                                      partition_coefficient always on raw.
                                      Header only if no foci were found.
  summary/all_ROIs.csv                all ROI tables combined
  summary/summary.csv                 per image (image_id): Otsu threshold,
                                      threshold used, number of foci,
                                      nucleus area (non-zero px), total
                                      foci area (px), foci area / nucleus area,
                                      nucleus and nucleoplasm mean intensity,
                                      fraction of nuclear signal in foci,
                                      size_gini (Gini of foci areas),
                                      mean_nn_distance (px), clark_evans_ratio
  binary_masks/<image>_mask.tif       foci mask (0/255)
  labeled_rois/<image>_labels.tif     labelled foci (1..N)
  backgrounds/, corrected_images/,    intermediate images (float32)
  smoothed_images/
  panels/<image>_six_panels.png       overview figure     (if save_figures = true)
  histograms/<image>_histograms.png   intensity histogram
                                      with the threshold  (if save_figures = true)
  run_config.toml                     copy of the parameter file used

next step:
  python agg.py   (input: <output>/roi_measurements) -> aggregated_data.csv

examples:
  python outsu.py -i nuclei_tif/ -o results/
  python outsu.py -i nuclei_tif/ -o results_t2/ -c config_threshold2.toml""",
    )
    parser.add_argument("-i", "--input", type=Path, metavar="DIR", required=True,
                        help="folder with the .tif/.tiff images to segment (required)")
    parser.add_argument("-o", "--output", type=Path, metavar="DIR", required=True,
                        help="folder for the results; created if it does not exist, "
                             "existing files with the same names are overwritten (required)")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG, metavar="FILE",
                        help="TOML file with the segmentation parameters; all keys are "
                             "required, see the comments in the file "
                             f"(default: {DEFAULT_CONFIG.name} next to this script)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    input_dir = args.input
    output_dir = args.output

    # Validate input folder
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    # Create output folder and subfolders (same layout as gmm.py)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dirs = {
        "histograms": output_dir / "histograms",
        "panels": output_dir / "panels",
        "background": output_dir / "backgrounds",
        "corrected": output_dir / "corrected_images",
        "smoothed": output_dir / "smoothed_images",
        "masks": output_dir / "binary_masks",
        "labels": output_dir / "labeled_rois",
        "measurements": output_dir / "roi_measurements",
        "summary": output_dir / "summary",
    }
    if not config["save_figures"]:
        del output_dirs["histograms"], output_dirs["panels"]

    for folder in output_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    # Keep the exact parameters used for this run next to the results
    shutil.copyfile(args.config, output_dir / "run_config.toml")

    # Find all TIFF images
    paths = sorted(
        list(input_dir.glob("*.tif")) +
        list(input_dir.glob("*.tiff"))
    )

    if not paths:
        raise FileNotFoundError(f"No .tif/.tiff images found in {input_dir}")

    print(f"Found {len(paths)} images")

    all_roi_tables, summary = [], []

    for path in paths:
        try:
            name = path.stem

            result = process_image(
                image_path=path,
                threshold_factor=config["threshold_factor"],
                sigma=config["sigma"],
                min_area=config["min_area"],
                min_distance=config["min_distance"],
                rolling_ball_radius=config["rolling_ball_radius"],
                segmentation_method=config["segmentation"],
            )

            # Figures
            if config["save_figures"]:
                plot_results(result, name, config,
                             output_dirs["panels"] / f"{name}_six_panels.png")
                draw_histogram(result, name, config["threshold_factor"],
                               output_dirs["histograms"] / f"{name}_histograms.png",
                               config["figure_dpi"])

            # Intermediate images (float32)
            for key in ("background", "corrected", "smoothed"):
                io.imsave(output_dirs[key] / f"{name}_{key}.tif",
                          result[key].astype(np.float32), check_contrast=False)

            # Binary mask and labelled ROIs
            io.imsave(output_dirs["masks"] / f"{name}_mask.tif",
                      result["mask"].astype(np.uint8) * 255, check_contrast=False)
            io.imsave(output_dirs["labels"] / f"{name}_labels.tif",
                      result["labels"].astype(np.int32), check_contrast=False)

            # ROI measurements
            rois = result["rois"]
            rois.to_csv(output_dirs["measurements"] / f"{name}_ROIs.csv", index=False)
            all_roi_tables.append(rois)

            # Foci pixels = pixels assigned to an ROI; fraction of the nucleus area
            foci_pixels = result["labels"] > 0
            nucleus_area = int(result["nucleus"].sum())
            summary.append({
                "image_id": name,
                "otsu_threshold": result["otsu_threshold"],
                "threshold_factor": config["threshold_factor"],
                "threshold": result["threshold"],
                "n_rois": len(rois),
                "nucleus_area_px": nucleus_area,
                "total_area_px": int(foci_pixels.sum()),
                "area_fraction": float(foci_pixels.sum() / nucleus_area),
                "nucleus_mean_intensity": result["metrics"]["nucleus_mean_intensity"],
                "nucleoplasm_mean_intensity": result["metrics"]["nucleoplasm_mean_intensity"],
                "fraction_signal_in_foci": result["metrics"]["fraction_signal_in_foci"],
                "size_gini": gini(rois["area"]),
                "mean_nn_distance": result["metrics"]["mean_nn_distance"],
                "clark_evans_ratio": result["metrics"]["clark_evans_ratio"],
            })

            print(f"Processed: {path.name} ({len(rois)} ROIs)")

        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            traceback.print_exc()
            continue
        finally:
            plt.close("all")

    # Combined results
    if all_roi_tables:
        pd.concat(all_roi_tables, ignore_index=True).to_csv(
            output_dirs["summary"] / "all_ROIs.csv", index=False
        )
    pd.DataFrame(summary).to_csv(output_dirs["summary"] / "summary.csv", index=False)

    print("Processing completed!")


if __name__ == "__main__":
    main()
