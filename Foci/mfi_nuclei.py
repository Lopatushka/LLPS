"""Nuclear area and mean fluorescence intensity (MFI) of single-nucleus TIFF images.

Run with -h for details.
"""
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def pixel_size(tif):
    """(pixel width, pixel height, unit) from the TIFF tags, as Fiji reads them.

    XResolution / YResolution = pixels per unit; the unit comes from the ImageJ
    metadata (e.g. "micron"). Uncalibrated images give (1, 1, "pixel").
    """
    page = tif.pages[0]
    unit = (tif.imagej_metadata or {}).get("unit", "")
    try:
        xres = page.tags["XResolution"].value
        yres = page.tags["YResolution"].value
        width, height = xres[1] / xres[0], yres[1] / yres[0]
    except (KeyError, ZeroDivisionError):
        return 1.0, 1.0, "pixel"
    if not unit or unit == "pixel" or (width == 1 and height == 1):
        return 1.0, 1.0, "pixel"
    return width, height, unit


def measure_image(path):
    """Area and mean intensity of the nucleus (pixels > 0) of one 2D image."""
    with tifffile.TiffFile(path) as tif:
        image = tif.asarray()
        width, height, unit = pixel_size(tif)

    if image.ndim != 2:
        raise ValueError(f"expected 2D grayscale TIFF, got shape {image.shape}")
    image = image.astype(np.float64)
    if not np.isfinite(image).all():
        raise ValueError("image contains NaN or infinity")

    # Nucleus = non-zero pixels (everything outside the nucleus is set to 0),
    # same definition as outsu.py / gmm.py
    nucleus = image > 0
    area_px = int(nucleus.sum())
    if area_px == 0:
        raise ValueError("image has no non-zero (nucleus) pixels")

    return {
        "image_id": path.stem,                  # same key as summary.csv of outsu.py / gmm.py
        "Filename": path.name,
        "Area": area_px * width * height,       # calibrated area (= Area_px if uncalibrated)
        "unit": f"{unit}^2" if unit != "pixel" else "px",
        "Area_px": area_px,
        "Mean": float(image[nucleus].mean()),   # = nucleus_mean_intensity of outsu.py / gmm.py
    }


def unique_path(path):
    """Return path, or <name>_1<ext>, <name>_2<ext>, ... if it already exists."""
    candidate, i = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        i += 1
    return candidate


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Area and mean fluorescence intensity (MFI) of the nucleus in single-nucleus
TIFF images. Nucleus = pixels > 0; the 0 pixels outside the nucleus are not
measured. Intensities are measured on the images as they are (raw, or
background-subtracted if you pass such images).""",
        epilog="""\
input:
  A folder with one 2D grayscale image per nucleus (*.tif / *.tiff),
  e.g. the single-nucleus images saved by FIJI/nuclei_manual_segment.py.

output:
  <output>/MFI_nuclei.csv   one row per image: image_id, Filename,
                            Area (calibrated, e.g. micron^2; = Area_px if the
                            image has no calibration), unit, Area_px (nucleus
                            area in pixels), Mean (mean nuclear intensity).
                            An existing MFI_nuclei.csv is never overwritten:
                            the table is saved as MFI_nuclei_1.csv, _2, ...

examples:
  python mfi_nuclei.py -i nuclei_tif/ -o results/""",
    )
    parser.add_argument("-i", "--input", type=Path, metavar="DIR", required=True,
                        help="folder with the .tif/.tiff images to measure (required)")
    parser.add_argument("-o", "--output", type=Path, metavar="DIR", required=True,
                        help="folder for MFI_nuclei.csv; created if it does not exist (required)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    paths = sorted(list(args.input.glob("*.tif")) + list(args.input.glob("*.tiff")))
    if not paths:
        raise FileNotFoundError(f"No .tif/.tiff images found in {args.input}")
    print(f"Found {len(paths)} images")

    rows = []
    for n, path in enumerate(paths, start=1):
        try:
            rows.append(measure_image(path))
            print(f"Processed {n}/{len(paths)}: {path.name}")
        except Exception as e:
            # The failed image gets no row in the table
            print(f"Error processing {path.name}: {e}")
            traceback.print_exc()

    out_path = unique_path(args.output / "MFI_nuclei.csv")
    pd.DataFrame(rows, columns=["image_id", "Filename", "Area", "unit", "Area_px", "Mean"]
                 ).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
