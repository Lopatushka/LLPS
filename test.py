from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage.draw import disk
from matplotlib.patches import Circle
from scipy.stats import spearmanr

def base_name_from_csv(filename: str) -> str:
    """
    Turn 'sample_0262-0212.csv' -> 'sample'
    Turn 'sample.csv' -> 'sample'
    """
    # remove extension
    name = re.sub(r"\.csv$", "", filename)
    # remove trailing _0262-0212 (or similar) if present
    name = re.sub(r"_\d+-\d+$", "", name)
    return name

def aggregate_data(dir1, dir2):
    # normalize paths (strip accidental spaces)
    dir_path1 = Path(str(dir1).strip()) # path to data about nucleus in total
    dir_path2 = Path(str(dir2).strip()) # path to data about foci in the particular nucleus

    if not dir_path1.exists():
        raise FileNotFoundError(f"dir1 not found: {dir_path1}")
    if not dir_path2.exists():
        raise FileNotFoundError(f"dir2 not found: {dir_path2}")
    
    # ---- NUCLEI TABLE (path1) ----
    files1 = sorted(dir_path1.glob("*.csv"))
    if not files1:
        raise FileNotFoundError(f"No CSV files found in dir1: {dir_path1}")
    
    dfs1 = [] 
    images_paths = []

    for f in files1:
        # find the corresponding image
        image_name = f.stem.replace("_roi", "") + ".jpg"
        image_path = f.with_name(image_name)
        images_paths.append(image_path)

        key = base_name_from_csv(f.name)
        key = key[:-4]
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()  # remove hidden spaces in headers

        # ensure expected columns exist
        if "Area" not in df.columns or "Mean" not in df.columns:
            raise KeyError(
                f"In nuclei file {f.name} expected columns 'Area' and 'Mean'. "
                f"Found: {list(df.columns)}"
            )
        
        df["File_name"] = key
        df = df.rename(columns={"Area": "Nucleus_area", "Mean": "Nucleus_MFI"})
        df = df[["File_name", "Nucleus_area", "Nucleus_MFI"]]
        dfs1.append(df)

    final = pd.concat(dfs1, ignore_index=True)

    print(images_paths)

    # ---- FOCI SUMMARY (path2) ----
    files2 = sorted(dir_path2.glob("*.csv"))
    if not files2:
        raise FileNotFoundError(f"No CSV files found in dir2: {dir_path2}")
    
    foci_rows = []

    check_column = lambda df, col: (
    df[col]
    if col in df.columns and not df.empty
    else pd.NA
    )
    
    for f in files2:
        print(f)
        key = base_name_from_csv(f.name)
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()


p1 = "/mnt/c/Users/Elena/Desktop/Data_processing/sb" 
p2 = "/mnt/c/Users/Elena/Desktop/Data_processing/sb/res" 
aggregate_data(p1, p2)