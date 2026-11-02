
import pandas as pd
import numpy as np
from pathlib import Path
import traceback


# =====================================================
# 1. AGGREGATE ONE IMAGE
# =====================================================

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

def main():

    input_path = input(
        "Enter input folder path [GMM_refined_segmentation/measurements]: "
    ).strip().strip('"')

    output_path = input(
        "Enter output folder path [GMM_refined_segmentation]: "
    ).strip().strip('"')

    input_dir = Path(
        input_path or "GMM_refined_segmentation/measurements"
    ).expanduser()

    output_dir = Path(
        output_path or "GMM_refined_segmentation"
    ).expanduser()

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
