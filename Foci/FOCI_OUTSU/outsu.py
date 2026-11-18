import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
import argparse
import shutil
import tomllib
from skimage import io, filters, morphology, segmentation, segmentation, exposure
from skimage.feature import peak_local_max
from skimage.restoration import rolling_ball, estimate_sigma
from scipy import ndimage as ndi
from skimage import io, filters, morphology, measure, segmentation, segmentation

from skimage.morphology import white_tophat, disk
    

def process_image(image_path,
                  threshold_factor=1,
                  sigma = 0.6,
                  min_area=2,
                  min_distance=3,
                  plot=True,
                  save_path=None,
                  figure_dpi=300):   
    
    # Download image
    image = io.imread(image_path)
    image_name = Path(image_path).stem
    
    # 1. Convert to float
    img = image.astype(np.float32)

    # 2. Background subtraction
    #background = rolling_ball(img, radius=20)
    #corrected = np.clip(img - background, 0, None)
    
    smoothed = filters.gaussian(
    img,
    sigma=sigma,
    preserve_range=True
    )
    
    threshold = filters.threshold_otsu(smoothed)
    
    binary_mask = smoothed > threshold * threshold_factor

    clean_mask = morphology.remove_small_objects(
        binary_mask,
        min_size=min_area
    )

    distance = ndi.distance_transform_edt(clean_mask)

    # Detect candidate centers
    coordinates = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=clean_mask
    )

    markers = np.zeros_like(clean_mask, dtype=np.int32)

    for i, (y, x) in enumerate(coordinates, start=1):
        markers[y, x] = i

    labels = segmentation.watershed(
        -distance,
        markers,
        mask=clean_mask
    )
    
    properties = measure.regionprops_table(
    labels,
    intensity_image=img,
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

    df = pd.DataFrame(properties)

    # Circularity: 1 means approximately circular
    df["circularity"] = (
        4 * np.pi * df["area"]
        / (df["perimeter"] ** 2 + 1e-8)
    )

    df["image_id"] = image_name


    #print("Total detected objects:", len(df))

    # Plot or save figure
    if plot or save_path is not None:

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].imshow(clean_mask, cmap="gray")
        axes[0].set_title("Binary mask")

        axes[1].imshow(distance, cmap="magma")
        axes[1].set_title("Distance transform")

        axes[2].imshow(
            segmentation.mark_boundaries(
                img / (img.max() + 1e-8),
                labels
            )
        )
        axes[2].set_title(f"Watershed: {labels.max()} objects")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        # Save figure
        if save_path is not None:
            save_path = Path(save_path)

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            fig.savefig(
                save_path,
                dpi=figure_dpi,
                bbox_inches="tight"
            )
            
    return df

DEFAULT_CONFIG = Path(__file__).with_name("config.toml")

CONFIG_TYPES = {
    "sigma": (int, float),
    "threshold_factor": (int, float),
    "min_area": int,
    "min_distance": int,
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

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Otsu + watershed foci segmentation of single-nucleus TIFF images.

For every image: Gaussian smoothing (sigma) -> mask = pixels above
Otsu threshold * threshold_factor -> removal of objects smaller than
min_area -> distance transform + watershed to split touching foci ->
shape and intensity measurements of every focus.""",
        epilog="""\
input:
  A folder with one 2D grayscale image per nucleus (*.tif / *.tiff),
  e.g. the single-nucleus images saved by FIJI/nuclei_manual_segment.py.

output (written to the --output folder):
  <image>_segmentation.csv   one row per focus: label, area, centroid-0/-1,
                             equivalent_diameter_area, eccentricity, solidity,
                             perimeter, mean/max/min_intensity, circularity,
                             image_id. Sizes are in pixels.
                             Header only if no foci were found.
  <image>_segmentation.png   binary mask, distance transform and watershed
                             overlay (only if save_figures = true)
  run_config.toml            copy of the parameter file used for this run

next step:
  python agg.py   (input: this output folder) -> aggregated_data.csv

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

    # Create output folder if necessary
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the exact parameters used for this run next to the results
    shutil.copyfile(args.config, output_dir / "run_config.toml")
    #print(f"Parameters from {args.config}: {config}")

    # Find all TIFF images
    paths = sorted(
        list(input_dir.glob("*.tif")) +
        list(input_dir.glob("*.tiff"))
    )

    print(f"Found {len(paths)} images")
    
    for path in paths:
        try:            
            # Output figure path
            save_path = output_dir / f"{path.stem}_segmentation.png"
            csv_path = output_dir / f"{path.stem}_segmentation.csv"
            
            result = process_image(
                image_path=path,
                threshold_factor=config["threshold_factor"],
                sigma=config["sigma"],
                min_area=config["min_area"],
                min_distance=config["min_distance"],
                plot=False,
                save_path=save_path if config["save_figures"] else None,
                figure_dpi=config["figure_dpi"],
                )
            plt.close("all")
            
            result.to_csv(csv_path, index=False)
            
            print(
                f"Processed: {path.name}"
            )

            
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            traceback.print_exc()
            continue
        
    print("Processing completed!")
    
if __name__ == "__main__":
    main()
