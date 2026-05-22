import os
import pandas as pd
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.draw import disk
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import math

def check_directory(path):
    if not isinstance(path, str):
        raise TypeError("Path must be a string.")
    
    path = path.strip()
    if path == "":
        raise ValueError("Path is empty.")
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Directory does not exist:\n{}".format(path)
        )

    if not os.path.isdir(path):
        raise NotADirectoryError(
            "Path is not a directory:\n{}".format(path)
        )

    return os.path.abspath(path)


def filename(path):
    """
    Return filename without extenstion
    """
    return os.path.splitext(os.path.basename(path))[0]


def nuclei_data(dir, output_dir):
    paths_csv_files = [
    os.path.join(dir, f)
    for f in os.listdir(dir)
    if os.path.isfile(os.path.join(dir, f))
    and f.lower().endswith("_roi.csv")
    ]

    # Number of .csv files and check
    n = len(paths_csv_files)
    if n == 0:
        raise FileNotFoundError(f"No CSV files found in the directory: {dir}")
    print(f"Founded {n} .csv files")

    # Create list of dataframes
    dfs = []
    for f in paths_csv_files:
        name = filename(f)
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()  # remove hidden spaces in headers

        # Ensure expected columns exist
        if "Area" not in df.columns or "Mean" not in df.columns:
            raise KeyError(
                f"In nuclei file {name} expected columns 'Area' and 'Mean'. "
                f"Found: {list(df.columns)}"
            )
        
        df["Filename"] = name # Add Filename column
        df = df.rename(columns={"Area": "Nucleus_area", "Mean": "Nucleus_MFI"}) # Rename columns
        df = df[["Filename", "Nucleus_area", "Nucleus_MFI"]] # change columns order
        dfs.append(df) # add df to the list of dataframes

    # Make final dataframe and export
    final = pd.concat(dfs, ignore_index=True)
    final.to_csv(f"{output_dir}/nuclei_results.csv", index=False)


def _sprearman_correlation(df):
    cols = df.select_dtypes(include="number").columns
    pairs = []

    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            x, y = df[c1], df[c2]
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n > 2:
                r, p = spearmanr(x[mask], y[mask])
                pairs.append({"var1": c1, "var2": c2, "n": n, "spearman_r": r, "p_value": p})

    pairs_df = pd.DataFrame(pairs)
    return pairs_df


def main():
    # Ask about paths with data and output directory to save results
    # Example of path: /mnt/c/users/elopatukhin/Desktop/Miscroscopy/160226_U2OS_fixed/MP_WT_0.3
    nuclei_dir = check_directory(input("Enter pathway to the directory with the information about nuclei (Area and Mean): "))
    while True:
        answer = input("Save results in the same folder as foci? (Y/N): ").strip().upper()
        if answer == "Y":
            output_dir = nuclei_dir
            break
        elif answer == "N":
            output_dir = check_directory(input("Enter output folder path: ").strip())
            break
        else:
            print("Please enter Y or N.")

    # --- Processed nuclei info (Area, Mean) ---
    nuclei_data(nuclei_dir, output_dir)

 
if __name__ == "__main__":
    main()


    
    