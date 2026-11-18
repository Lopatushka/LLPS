import pandas as pd
from pathlib import Path
import traceback
import argparse

def table_aggregation(path):
    df = pd.read_csv(path)

    # Cells with zero foci produce a header-only CSV: take the ID from the
    # filename instead of dropping the cell (outsu.py saves <stem>_segmentation.csv)
    if len(df) and "image_id" in df.columns:
        image_id = df["image_id"].iloc[0]
    else:
        image_id = path.stem.removesuffix("_segmentation")
    n_foci = df.shape[0]
    
    area = df["area"].median()
    equivalent_diameter_area = df["equivalent_diameter_area"].median()
    eccentricity = df["eccentricity"].median()
    solidity = df["solidity"].median()
    perimeter = df["perimeter"].median()
    mean_intensity = df["mean_intensity"].median()
    max_intensity = df["max_intensity"].median()
    min_intensity = df["min_intensity"].median()
    
    row = {
        "image_id": image_id,
        "n_foci": n_foci,
        "area": area,
        "equivalent_diameter_area": equivalent_diameter_area,
        "eccentricity": eccentricity,
        "solidity": solidity,
        "perimeter": perimeter,
        "mean_intensity": mean_intensity,
        "max_intensity": max_intensity,
        "min_intensity": min_intensity,
        
    }
    
    return row

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Aggregate the per-image foci tables written by outsu.py into one table
with one row per cell (image).

For every <image>_segmentation.csv: n_foci = number of foci, plus the
median of each foci feature over all foci of that cell. Cells without
foci are kept with n_foci = 0 and empty (NaN) feature values.""",
        epilog="""\
input:
  The output folder of outsu.py, containing <image>_segmentation.csv files
  (other files, including an earlier aggregated_data.csv, are ignored).

output:
  <output>/aggregated_data.csv   one row per cell with columns:
                                 image_id, n_foci, and the medians of area,
                                 equivalent_diameter_area, eccentricity,
                                 solidity, perimeter, mean_intensity,
                                 max_intensity, min_intensity.
                                 Sizes are in pixels.

next step:
  Add the aggregated_data.csv of each condition to the `data` list in
  STATISTICS/PCA.ipynb or STATISTICS/UMAP.ipynb.

examples:
  python agg.py -i results/ -o results/
  python agg.py -i WT/otsu/ -o WT/""",
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
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    
    # Create output folder if necessary
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find per-image foci tables written by outsu.py (excludes aggregated_data.csv)
    paths = sorted(input_dir.glob("*_segmentation.csv"))

    if not paths:
        raise FileNotFoundError(f"No *_segmentation.csv files found in {input_dir}")

    print(f"Found {len(paths)} csv files")
    
    rows = []
    for path in paths:
        try:            
            row = table_aggregation(path)
            rows.append(row)
            
            print(
                f"Processed: {path.name}"
            )
            
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            traceback.print_exc()
            continue
        
    final = pd.DataFrame(rows)
    csv_path = output_dir / f"aggregated_data.csv"
    final.to_csv(csv_path, index=False)
    
    print("Processing completed!")
    
if __name__ == "__main__":
    main()
            