
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
import argparse


# =====================================================
# 1. AGGREGATE ONE IMAGE
# =====================================================

# Per-cell values copied from <output>/summary/summary.csv
CELL_FEATURES = ["fraction_signal_in_foci", "size_gini"]

def table_aggregation(path):

    df = pd.read_csv(path)

    # ----------------------------------------------
    # IMAGE IDENTIFIER
    # ----------------------------------------------

    if "image" in df.columns:
        image_id = df["image"].iloc[0] if len(df) else path.stem.removesuffix("_ROIs")

    elif "image_id" in df.columns:
        image_id = df["image_id"].iloc[0] if len(df) else path.stem.removesuffix("_ROIs")

    else:
        image_id = path.stem.removesuffix("_ROIs")

    # Older gmm.py runs stored the image name with its extension (cell0.tif):
    # strip it so image_id is the same with and without foci, and as in outsu.py
    image_id = str(image_id)
    for ext in (".tiff", ".tif", ".TIFF", ".TIF"):
        image_id = image_id.removesuffix(ext)

    # ----------------------------------------------
    # NUMBER OF FOCI
    # ----------------------------------------------

    n_foci = len(df)

    # ----------------------------------------------
    # FEATURES TO AGGREGATE
    # ----------------------------------------------

    features = [
        "area",
        "equivalent_diameter_area",
        "eccentricity",
        "solidity",
        "perimeter",
        "mean_intensity",
        "max_intensity",
        "min_intensity",
        "partition_coefficient",
    ]

    row = {
        "image_id": image_id,
        "n_foci": n_foci
    }

    # ----------------------------------------------
    # MEDIAN OF EACH AVAILABLE FEATURE
    # ----------------------------------------------

    for feature in features:

        if feature in df.columns:

            row[feature] = pd.to_numeric(
                df[feature],
                errors="coerce"
            ).median()

        else:
            row[feature] = np.nan

    return row


# =====================================================
# 2. PROCESS ALL CSV FILES
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Aggregate the per-image ROI tables written by outsu.py or
gmm.py into one table
with one row per cell (image).

For every <image>_ROIs.csv: n_foci = number of foci, plus the median of
each foci feature over all foci of that cell. Cells without foci are
kept with n_foci = 0 and empty (NaN) feature values.""",
        epilog="""\
input:
  The roi_measurements/ subfolder of an outsu.py or gmm.py output folder,
  containing
  <image>_ROIs.csv files (other files are ignored).

output:
  <output>/aggregated_data.csv   one row per cell with columns:
                                 image_id, n_foci, and the medians of area,
                                 equivalent_diameter_area, eccentricity,
                                 solidity, perimeter, mean_intensity,
                                 max_intensity, min_intensity,
                                 partition_coefficient, plus the per-cell
                                 fraction_signal_in_foci and size_gini (read from
                                 summary/summary.csv next to the input folder).
                                 Sizes are in pixels.

next step:
  Add the aggregated_data.csv of each condition to the `data` list in
  STATISTICS/PCA.ipynb or STATISTICS/UMAP.ipynb.

examples:
  python agg.py -i results_otsu/roi_measurements/ -o results_otsu/
  python agg.py -i WT/gmm4/roi_measurements/ -o WT/gmm4/""",
    )
    parser.add_argument("-i", "--input", type=Path, metavar="DIR", required=True,
                        help="folder with the per-image foci tables (required)")
    parser.add_argument("-o", "--output", type=Path, metavar="DIR", required=True,
                        help="folder for aggregated_data.csv; created if it does not exist, "
                             "an existing aggregated_data.csv is overwritten (required)")
    return parser.parse_args()


def main():

    args = parse_args()

    input_dir = args.input
    output_dir = args.output

    # Validate input folder
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_dir}"
        )

    # Create output folder
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------
    # FIND INDIVIDUAL ROI TABLES
    # ----------------------------------------------

    paths = sorted(
        input_dir.glob("*_ROIs.csv")
    )

    if not paths:
        raise FileNotFoundError(
            f"No *_ROIs.csv files found in {input_dir}"
        )

    print(f"\nFound {len(paths)} ROI tables")

    # ----------------------------------------------
    # AGGREGATE EACH IMAGE
    # ----------------------------------------------

    rows = []

    for path in paths:

        try:

            row = table_aggregation(path)

            rows.append(row)

            print(
                f"Processed: {path.name} "
                f"({row['n_foci']} ROIs)"
            )

        except Exception as e:

            print(
                f"Error processing {path.name}: {e}"
            )

            traceback.print_exc()

    # ----------------------------------------------
    # COMBINE RESULTS
    # ----------------------------------------------

    if not rows:
        raise ValueError(
            "No ROI tables were successfully processed"
        )

    final = pd.DataFrame(rows)

    # ----------------------------------------------
    # PER-CELL VALUES FROM summary/summary.csv
    # ----------------------------------------------
    # The fraction of nuclear signal in foci is one value per cell, written by
    # outsu.py / gmm.py into <output>/summary/summary.csv next to roi_measurements/

    summary_path = input_dir.parent / "summary" / "summary.csv"
    if summary_path.is_file():
        per_cell = pd.read_csv(summary_path)
        per_cell["image_id"] = per_cell["image_id"].astype(str)
        present = [c for c in CELL_FEATURES if c in per_cell.columns]
        missing = [c for c in CELL_FEATURES if c not in per_cell.columns]
        if present:
            final["image_id"] = final["image_id"].astype(str)
            final = final.merge(per_cell[["image_id", *present]],
                                on="image_id", how="left")
            print(f"Added {', '.join(present)} from {summary_path}")
        if missing:
            print(f"Warning: {summary_path} has no {', '.join(missing)} column(s) "
                  "(older run?); not added")
    else:
        print(f"Warning: {summary_path} not found; per-cell features not added")

    # ----------------------------------------------
    # SAVE
    # ----------------------------------------------

    csv_path = output_dir / "aggregated_data.csv"

    final.to_csv(
        csv_path,
        index=False
    )

    print("\nProcessing completed!")
    print(f"Images aggregated: {len(final)}")
    print(f"Saved to: {csv_path.resolve()}")

    #return final


if __name__ == "__main__":
    main()
