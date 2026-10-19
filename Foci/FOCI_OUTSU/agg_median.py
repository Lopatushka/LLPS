import pandas as pd
from pathlib import Path
import traceback

def table_aggregation(path):
    df = pd.read_csv(path)
    
    image_id = df["image_id"][0]
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

def main():
    input_dir = Path(input("Enter input folder path: ").strip().strip('"'))
    output_dir = Path(input("Enter output folder path: ").strip().strip('"'))
    
    # Validate input folder
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    
    # Create output folder if necessary
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all TIFF images
    paths = sorted(
        list(input_dir.glob("*.csv")))

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
            