"""GMM foci segmentation of 2D TIFF images.

Install: pip install -r requirements.txt
Run:     python gmm.py --input <images> --output <results> [--config config_gmm.toml]

Parameters are read from config_gmm.toml (next to this script by default) and a copy
is saved into the output folder as run_config.toml.
"""
from pathlib import Path
import argparse
import shutil
import traceback
import tomllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_fill_holes, distance_transform_edt
from scipy import ndimage as ndi
from scipy.stats import norm
from skimage import io, measure, morphology, restoration, segmentation
from skimage.feature import peak_local_max
from sklearn.mixture import GaussianMixture
import time

# ========== DEFAULTS ==========
# Used when the module is imported. main() overrides them from config.toml.
N_COMPONENTS = 4 # Number of GMM components to fit
SIGNAL_COMPONENTS = [4] # Components selected for segmentation (ordered by mean intensity)

PROBABILITY_THRESHOLD = 0.5
ROLLING_BALL_RADIUS = 0 # 0 = skip background subtraction
SMOOTH_SIGMA = 1.0
MIN_AREA = 20
FILL_HOLES = True
N_INIT = 10
RANDOM_STATE = 42
HIST_BINS = 200
HIST_PERCENTILE = 99.99
FIG_DPI = 180
MAX_FIT_PIXELS = 150000 # Subsample size for the GMM fit; 0 = use all pixels.
SEGMENTATION = "watershed" # How the mask is split into ROIs: "watershed" or "connected"
MIN_DISTANCE = 3 # Minimum distance between watershed seeds, in pixels
SEGMENTATION_METHODS = ("watershed", "connected")


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

def log_step(message, start):
    print(
        f"{message}: {time.perf_counter() - start:.2f} s",
        flush=True
    )
    return time.perf_counter()


def validate_parameters():
    if (
        not SIGNAL_COMPONENTS
        or len(set(SIGNAL_COMPONENTS)) != len(SIGNAL_COMPONENTS)
        or any(c < 1 or c > N_COMPONENTS for c in SIGNAL_COMPONENTS)
        ):
        raise ValueError(
            "SIGNAL_COMPONENTS must contain unique component numbers "
            "between 1 and N_COMPONENTS"
        )
    
    if not (0 < PROBABILITY_THRESHOLD <= 1):
        raise ValueError('PROBABILITY_THRESHOLD must be in (0, 1]')
    
    if ROLLING_BALL_RADIUS is not None and ROLLING_BALL_RADIUS < 0:
        raise ValueError('ROLLING_BALL_RADIUS must be >= 0 (0 = no background subtraction)')
    
    if MAX_FIT_PIXELS is not None and MAX_FIT_PIXELS < 0:
        raise ValueError('MAX_FIT_PIXELS must be >= 0 (0 = use all pixels)')
        
    if SEGMENTATION not in SEGMENTATION_METHODS:
        raise ValueError(f'SEGMENTATION must be one of {SEGMENTATION_METHODS}')

    if MIN_DISTANCE < 1:
        raise ValueError('MIN_DISTANCE must be >= 1')

    if SMOOTH_SIGMA < 0 or MIN_AREA < 1:
        raise ValueError('SMOOTH_SIGMA must be >= 0 and MIN_AREA >= 1')


def process_image(path):
    #t = time.perf_counter()
     
    original = io.imread(path)
    if original.ndim != 2:
        raise ValueError(f'{path.name}: expected 2D grayscale TIFF, got {original.shape}')
    
    original = original.astype(np.float64)
    if not np.isfinite(original).all():
        raise ValueError(f'{path.name}: image contains NaN or infinity')

    # Nucleus = non-zero pixels (everything outside the nucleus is set to 0)
    nucleus = original > 0
    if not nucleus.any():
        raise ValueError(f'{path.name}: image has no non-zero (nucleus) pixels')
    
    #t = log_step("Image loaded", t)
    
    # Background subtraction and smoothing
    if not ROLLING_BALL_RADIUS:
        background = np.zeros_like(original)
        corrected = original.copy()
    else:
        background = restoration.rolling_ball(original, radius=ROLLING_BALL_RADIUS)
        #t = log_step("Rolling ball completed", t)
        corrected = np.maximum(original - background, 0)
        
    smoothed = gaussian_filter(corrected, sigma=SMOOTH_SIGMA)
    
    #t = log_step("Smoothing completed", t)
    
    pixels = smoothed.reshape(-1, 1)
    
    if MAX_FIT_PIXELS and len(pixels) > MAX_FIT_PIXELS:
        rng = np.random.default_rng(RANDOM_STATE)
        fit_pixels = pixels[rng.choice(len(pixels), MAX_FIT_PIXELS, replace=False)]
    else:
        fit_pixels = pixels
        
    if len(fit_pixels) < N_COMPONENTS:
        raise ValueError('Not enough pixels for the requested GMM components')
    
    gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type='full',
                          n_init=N_INIT, random_state=RANDOM_STATE, reg_covar=1e-6)
    gmm.fit(fit_pixels)
    
    #t = log_step("GMM fitting completed", t)
     
    #order = np.argsort(gmm.means_.ravel())
    #signal_indices = order[-SIGNAL_COMPONENTS:]
    
    order = np.argsort(gmm.means_.ravel())
    # Convert component numbers (1-based) to actual GMM indices
    signal_indices = order[
        np.array(SIGNAL_COMPONENTS) - 1
    ]
    
    # Chunk posterior calculation to avoid large allocations for big images.
    signal_prob = np.empty(len(pixels), dtype=np.float32)
    for start in range(0, len(pixels), 250000):
        stop = min(start + 250000, len(pixels))
        signal_prob[start:stop] = gmm.predict_proba(pixels[start:stop])[:, signal_indices].sum(axis=1)
    
    signal_prob = signal_prob.reshape(original.shape)
    
    #t = log_step("Probabilities completed", t)
    
    mask = (signal_prob >= PROBABILITY_THRESHOLD) & nucleus
    mask = morphology.remove_small_objects(mask, min_size=MIN_AREA, connectivity=2)
    
    if FILL_HOLES:
        mask = binary_fill_holes(mask)
        
    if SEGMENTATION == "watershed":
        # Distance transform + watershed to split touching foci
        distance = distance_transform_edt(mask)
        coordinates = peak_local_max(distance, min_distance=MIN_DISTANCE, labels=mask)
        markers = np.zeros(mask.shape, dtype=np.int32)
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
        # connectivity=2: also reach diagonally connected pixels
        labels = segmentation.watershed(-distance, markers, mask=mask, connectivity=2)

        # Merge fragments smaller than min_area into the touching ROI: the
        # remaining ROIs are grown back over the mask by a second watershed.
        # Fragments that touch no larger ROI stay 0 and are removed below.
        big = morphology.remove_small_objects(labels, min_size=MIN_AREA)
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
    labels = morphology.remove_small_objects(labels, min_size=MIN_AREA)
    labels = segmentation.relabel_sequential(labels)[0]
    mask = labels > 0
    
    #t = log_step("Segmentation completed", t)
    
    props = measure.regionprops_table(labels, intensity_image=corrected,  # background-subtracted if ROLLING_BALL_RADIUS > 0
        properties=('label', 'area', 'centroid', 'equivalent_diameter_area',
                    'eccentricity', 'solidity', 'perimeter', 'mean_intensity', 'max_intensity', 'min_intensity'))
    
    rois = pd.DataFrame(props)
    # Image name without extension, same column as in outsu.py
    rois.insert(0, 'image_id', path.stem)

    # Partition coefficient and fraction of signal in foci (raw image)
    metrics = signal_metrics(original, nucleus, labels)
    rois['partition_coefficient'] = metrics['partition_coefficient'][rois['label'].to_numpy(int) - 1]
    
    #t = log_step("ROI measurements completed", t)
    
    return dict(metrics=metrics, original=original, nucleus=nucleus, background=background, corrected=corrected,
                smoothed=smoothed, mask=mask, labels=labels, gmm=gmm,
                signal_indices=signal_indices, signal_prob=signal_prob, rois=rois)


def histogram_data(image, gmm, bins=None, percentile=None):
    # Read the module settings at call time so that config.toml values apply
    bins = HIST_BINS if bins is None else bins
    percentile = HIST_PERCENTILE if percentile is None else percentile
    values = image.ravel()
    lo = float(values.min())
    hi = float(np.percentile(values, percentile))
    if hi <= lo:
        hi = float(values.max())
    if hi <= lo:
        hi = lo + 1
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    density = counts / (len(values) * np.diff(edges))
    x = np.linspace(lo, hi, 1500)
    means = gmm.means_.ravel()
    stds = np.sqrt(gmm.covariances_.reshape(-1))
    weights = gmm.weights_
    order = np.argsort(means)
    pdfs = {int(k): weights[k] * norm.pdf(x, means[k], stds[k]) for k in order}
    total = sum(pdfs.values())
    return edges, density, x, order, pdfs, total, means, stds, weights

def _draw_histogram_ax(ax, hist, signal_indices, log=False, legend=True):
    edges, density, x, order, pdfs, total, means, stds, weights = hist

    ax.stairs(
        density, edges,
        fill=True,
        color="gray",
        alpha=0.32,
        label="Pixel intensities"
    )

    colors = plt.get_cmap("tab10")
    signal_set = set(map(int, signal_indices))

    for rank, k in enumerate(order):
        selected = int(k) in signal_set

        ax.plot(
            x,
            pdfs[int(k)],
            color=colors(rank % 10),
            linewidth=2.2 if selected else 1.3,
            linestyle="-" if selected else "--",
            label=(
                f"G{rank+1}: μ={means[k]:.2f}, "
                f"σ={stds[k]:.2f}, w={weights[k]:.3f}"
                + (" [SIGNAL]" if selected else "")
            )
        )

    ax.plot(x, total, color="black", lw=2.3, label="Total GMM")

    if log:
        ax.set_yscale("log")
        positives = np.r_[density[density > 0], total[total > 0]]

        if len(positives):
            ax.set_ylim(
                bottom=max(float(np.min(positives)) * 0.5, 1e-12)
            )

    ax.set_title(
        "Histogram + GMM (" + ("log Y" if log else "linear Y") + ")"
    )
    ax.set_xlabel("Smoothed, background-corrected intensity")
    ax.set_ylabel("Probability density")
    ax.grid(alpha=0.2)

    if legend:
        ax.legend(fontsize=7)


def draw_histogram(
    result,
    name=None,
    save_path=None,
    figsize=(8, 5),
    show=False
):
    """
    Create a log-scale GMM histogram.

    If save_path is provided, save the figure.
    If show=True, display the figure.
    Always close the figure afterward.
    """
    hist = histogram_data(
        result["smoothed"],
        result["gmm"]
    )

    fig, ax = plt.subplots(figsize=figsize)

    _draw_histogram_ax(
        ax,
        hist,
        result["signal_indices"],
        log=True
    )

    if name is not None:
        fig.suptitle(f"{name}: GMM intensity distributions", fontsize=14)

    fig.tight_layout()

    try:
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_path,
                dpi=FIG_DPI,
                bbox_inches="tight"
            )

        if show:
            plt.show()
    finally:
        plt.close(fig)

def plot_results(result, name, output=None):
    """
    Create a six-panel visualization of GMM segmentation.

    If output is provided, save the figure.
    The figure is automatically closed after saving/displaying.
    """

    original = result["original"]
    background = result["background"]
    corrected = result["corrected"]
    smoothed = result["smoothed"]
    mask = result["mask"]
    labels = result["labels"]

    hist = histogram_data(smoothed, result["gmm"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # First four panels
    images = [
        original,
        background,
        corrected,
        mask
    ]

    titles = [
        "Original image",
        f"Rolling-ball background (r={ROLLING_BALL_RADIUS})",
        "Background corrected (before smoothing)",
        "Binary mask (filled holes)"
    ]

    for ax, img, title in zip(axes.flat[:4], images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    # Fifth panel: Segmented ROIs
    ax = axes[1, 1]
    # ROI boundaries (shows the watershed splits)
    span = original.max() - original.min()
    ax.imshow(segmentation.mark_boundaries(
        (original - original.min()) / (span if span > 0 else 1),
        labels, color=(0, 1, 0)
    ))

    for region in measure.regionprops(labels):
        y, x = region.centroid

        ax.text(
            x, y,
            str(region.label),
            color="yellow",
            fontsize=6,
            ha="center",
            va="center"
        )

    ax.set_title(f"Segmented ROIs ({SEGMENTATION}, N={labels.max()})")
    ax.axis("off")

    # Sixth panel: GMM histogram
    _draw_histogram_ax(
        axes[1, 2],
        hist,
        result["signal_indices"],
        log=True
    )

    # Figure title
    selected = ", ".join(f"G{i}" for i in SIGNAL_COMPONENTS)
    
    fig.suptitle(
        f"{name} | K={N_COMPONENTS} | Selected: {selected}, "
        f"σ={SMOOTH_SIGMA}",
        fontsize=15
    )

    fig.tight_layout()

    # Optional saving
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            output / f"{name}_six_panels.png",
            dpi=FIG_DPI,
            bbox_inches="tight"
        )

    plt.close(fig)


DEFAULT_CONFIG = Path(__file__).with_name("config_gmm.toml")

# config.toml key -> (module variable, allowed types)
CONFIG_KEYS = {
    "n_components": ("N_COMPONENTS", int),
    "signal_components": ("SIGNAL_COMPONENTS", list),
    "probability_threshold": ("PROBABILITY_THRESHOLD", (int, float)),
    "rolling_ball_radius": ("ROLLING_BALL_RADIUS", (int, float)),
    "smooth_sigma": ("SMOOTH_SIGMA", (int, float)),
    "min_area": ("MIN_AREA", int),
    "fill_holes": ("FILL_HOLES", bool),
    "segmentation": ("SEGMENTATION", str),
    "min_distance": ("MIN_DISTANCE", int),
    "n_init": ("N_INIT", int),
    "random_state": ("RANDOM_STATE", int),
    "max_fit_pixels": ("MAX_FIT_PIXELS", int),
    "hist_bins": ("HIST_BINS", int),
    "hist_percentile": ("HIST_PERCENTILE", (int, float)),
    "figure_dpi": ("FIG_DPI", int),
}


def load_config(path):
    """Read the TOML config, check it and apply it to the module settings."""
    with open(path, "rb") as f:
        config = tomllib.load(f)

    missing = CONFIG_KEYS.keys() - config.keys()
    unknown = config.keys() - CONFIG_KEYS.keys()
    if missing or unknown:
        raise ValueError(
            f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}"
        )

    for key, (name, types) in CONFIG_KEYS.items():
        value = config[key]
        # bool is a subclass of int: reject true/false for numeric keys
        if not isinstance(value, types) or (types is not bool and isinstance(value, bool)):
            raise TypeError(f"{path}: '{key}' has invalid value {value!r}")
        if key == "signal_components" and not all(
            isinstance(c, int) and not isinstance(c, bool) for c in value
        ):
            raise TypeError(f"{path}: 'signal_components' must be a list of integers")
        globals()[name] = value

    validate_parameters()
    if N_INIT < 1 or HIST_BINS < 1 or FIG_DPI < 1:
        raise ValueError(f"{path}: n_init, hist_bins and figure_dpi must be >= 1")
    if not (0 < HIST_PERCENTILE <= 100):
        raise ValueError(f"{path}: hist_percentile must be in (0, 100]")

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Gaussian-mixture-model (GMM) foci segmentation of 2D TIFF images.

For every image: optional rolling-ball background subtraction ->
Gaussian smoothing (smooth_sigma) -> GMM with n_components fitted to the
pixel intensities -> pixels whose probability of belonging to the
signal_components is >= probability_threshold form the foci mask ->
removal of objects smaller than min_area, optional hole filling ->
foci ROIs (segmentation = "watershed": distance transform + watershed to
split touching foci; "connected": connected regions) -> shape and intensity measurements of every focus.""",
        epilog="""\
input:
  A folder with 2D grayscale images (*.tif / *.tiff), one nucleus per image,
  e.g. the single-nucleus images saved by FIJI/nuclei_manual_segment.py.
  Other files in the folder are ignored.

output (subfolders of the --output folder):
  roi_measurements/<image>_ROIs.csv   one row per focus: image_id, label,
                                      area, centroid-0/-1,
                                      equivalent_diameter_area, eccentricity,
                                      solidity, perimeter,
                                      mean/max/min_intensity,
                                      partition_coefficient. Sizes are in
                                      pixels.
                                      Intensities are measured on the
                                      background-subtracted image if
                                      rolling_ball_radius > 0, else raw;
                                      partition_coefficient always on raw.
  summary/all_ROIs.csv                all ROI tables combined
  summary/summary.csv                 per image (image_id): number of foci,
                                      nucleus area (non-zero px), total foci
                                      area (px), foci area / nucleus area,
                                      nucleus and nucleoplasm mean intensity,
                                      fraction of nuclear signal in foci,
                                      size_gini (Gini of foci areas)
  binary_masks/<image>_mask.tif       foci mask (0/255)
  labeled_rois/<image>_labels.tif     labelled foci (1..N)
  backgrounds/, corrected_images/,    intermediate images (float32)
  smoothed_images/
  panels/<image>_six_panels.png       overview figure
  histograms/<image>_histograms.png   intensity histogram with fitted GMM
  run_config.toml                     copy of the parameter file used

next step:
  python agg.py   (input: <output>/roi_measurements) -> aggregated_data.csv

examples:
  python gmm.py -i nuclei_tif/ -o results_gmm/
  python gmm.py -i nuclei_tif/ -o results_g34/ -c config_g34.toml""",
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
    
    IMAGE_FOLDER = args.input
    OUTPUT_FOLDER = args.output
    
    # Validate input directory
    if not IMAGE_FOLDER.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {IMAGE_FOLDER}"
        )
        
    # Create output directory if necessary
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )
    
    # Create output subfolders
    OUTPUT_DIRS = {
        "histograms": OUTPUT_FOLDER / "histograms",
        "panels": OUTPUT_FOLDER / "panels",
        "backgrounds": OUTPUT_FOLDER / "backgrounds",
        "corrected": OUTPUT_FOLDER / "corrected_images",
        "smoothed": OUTPUT_FOLDER / "smoothed_images",
        "masks": OUTPUT_FOLDER / "binary_masks",
        "labels": OUTPUT_FOLDER / "labeled_rois",
        "measurements": OUTPUT_FOLDER / "roi_measurements",
        "summary": OUTPUT_FOLDER / "summary",
    }
    
    for folder in OUTPUT_DIRS.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    # Keep the exact parameters used for this run next to the results
    shutil.copyfile(args.config, OUTPUT_FOLDER / "run_config.toml")
    print(f"Parameters from {args.config}: {config}")
    
    print(f"\nInput directory: {IMAGE_FOLDER.resolve()}")
    print(f"Output directory: {OUTPUT_FOLDER.resolve()}\n")
    
    # Check images in the input directory
    paths = sorted(p for p in IMAGE_FOLDER.iterdir() if p.is_file() and p.suffix.lower() in ('.tif', '.tiff'))
    if len(paths) == 0:
        raise ValueError(f'No TIFF images found in {IMAGE_FOLDER.resolve()}')
    
    all_roi_tables, summary = [], []
    
    # Import images and process them; an error in one image skips it
    failed = []
    for path in paths:
        try:
            print(f"\nProcessing: {path.name}", flush=True)
        
            results = process_image(path)
            name = path.stem
        
            # Six-panel visualization
            plot_results(
                results,
                name=name,
                output=OUTPUT_DIRS["panels"]
            )
        
            # Separate GMM histogram
            draw_histogram(
                results,
                name=name,
                save_path=OUTPUT_DIRS["histograms"] / f"{name}_histograms.png"
            )

            # Background image
            io.imsave(
                OUTPUT_DIRS["backgrounds"] / f"{name}_background.tif",
                results["background"].astype(np.float32),
                check_contrast=False
            )

            # Background-corrected image
            io.imsave(
                OUTPUT_DIRS["corrected"] / f"{name}_corrected.tif",
                results["corrected"].astype(np.float32),
                check_contrast=False
            )

            # Smoothed image
            io.imsave(
                OUTPUT_DIRS["smoothed"] / f"{name}_smoothed.tif",
                results["smoothed"].astype(np.float32),
                check_contrast=False
            )

            # Binary mask
            io.imsave(
                OUTPUT_DIRS["masks"] / f"{name}_mask.tif",
                results["mask"].astype(np.uint8) * 255,
                check_contrast=False
            )

            # Labeled ROIs
            io.imsave(
                OUTPUT_DIRS["labels"] / f"{name}_labels.tif",
                results["labels"].astype(np.int32),
                check_contrast=False
            )

            # ROI measurements
            roi_df = results["rois"].copy()

            roi_df.to_csv(
                OUTPUT_DIRS["measurements"] / f"{name}_ROIs.csv",
                index=False
            )

            all_roi_tables.append(roi_df)

            summary.append({
                "image_id": path.stem,
                "n_components": N_COMPONENTS,
                "signal_components": SIGNAL_COMPONENTS,
                "n_rois": int(results["labels"].max()),
                "nucleus_area_px": int(results["nucleus"].sum()),
                "total_area_px": int(results["mask"].sum()),
                "area_fraction": float(results["mask"].sum() / results["nucleus"].sum()),
                "nucleus_mean_intensity": results["metrics"]["nucleus_mean_intensity"],
                "nucleoplasm_mean_intensity": results["metrics"]["nucleoplasm_mean_intensity"],
                "fraction_signal_in_foci": results["metrics"]["fraction_signal_in_foci"],
                "size_gini": gini(roi_df["area"]),
            })

        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            traceback.print_exc()
            failed.append(path.name)
        finally:
            plt.close("all")

    # Save combined results
    if all_roi_tables:
        combined_df = pd.concat(all_roi_tables, ignore_index=True)

        combined_df.to_csv(
            OUTPUT_DIRS["summary"] / "all_ROIs.csv",
            index=False
        )

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        OUTPUT_DIRS["summary"] / "summary.csv",
        index=False
    )

    if failed:
        print(f"\nDone with errors: {len(failed)} of {len(paths)} images failed: {', '.join(failed)}")
    else:
        print("\nAll images processed successfully.")
       

if __name__ == '__main__':
    main()