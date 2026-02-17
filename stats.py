from pathlib import Path
import re
import pandas as pd
import numpy as np

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.color import rgb2gray
from skimage.draw import disk

from scipy.stats import spearmanr
#from scipy.stats import ttest_ind
#from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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
    dir_path1 = Path(str(dir1).strip())
    dir_path2 = Path(str(dir2).strip())

    if not dir_path1.exists():
        raise FileNotFoundError(f"dir1 not found: {dir_path1}")
    if not dir_path2.exists():
        raise FileNotFoundError(f"dir2 not found: {dir_path2}")
    
    # ---- NUCLEI TABLE (path1) ----
    files1 = sorted(dir_path1.glob("*.csv"))
    if not files1:
        raise FileNotFoundError(f"No CSV files found in dir1: {dir_path1}")
    
    dfs1 = []
    for f in files1:
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

    # ---- FOCI SUMMARY (path2) ----
    files2 = sorted(dir_path2.glob("*.csv"))
    if not files2:
        raise FileNotFoundError(f"No CSV files found in dir2: {dir_path2}")
    
    foci_rows = []
    for f in files2:
        key = base_name_from_csv(f.name)
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        # count rows + mean intensity
        foci_rows.append({
        "File_name": key,
        "Foci_number": int(df.shape[0]),
        "Foci_MFI": float(df["intensity [photon]"].mean()) if "intensity [photon]" in df.columns and df.shape[0] > 0 else pd.NA,
        "Foci_sigma": float(df["sigma [nm]"].mean()) if "sigma [nm]" in df.columns and df.shape[0] > 0 else pd.NA
    })

    foci_summary = pd.DataFrame(foci_rows)

    # ---- MERGE ----
    final["File_name"] = final["File_name"].astype(str).str.strip()
    foci_summary["File_name"] = foci_summary["File_name"].astype(str).str.strip()

    merged  = final.merge(foci_summary, on="File_name", how="left")
    merged["Foci_MFI"] = pd.to_numeric(merged["Foci_MFI"], errors="coerce")

    return merged

def sprearman_correlation(df):
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

def compute_mean_intensity_from_localizations(
        path_image,
        df,
        orig_size=(2560, 2560),
        pixel_size_original_nm=16.0,
        x_col="x [nm]",
        y_col="y [nm]",
        sigma_col="sigma [nm]"
    ):
    """
    Convert ThunderSTORM localizations (nm) into current image pixels,
    build circular ROIs, and compute mean grayscale intensity inside each ROI.

    Parameters
    ----------
    image : PIL Image or numpy array
        RGB image on which intensity will be measured.
    df : pandas DataFrame
        Must contain x, y, sigma columns in nanometers.
    orig_size : tuple
        Original image size used during ThunderSTORM analysis (width, height).
    pixel_size_original_nm : float
        Pixel size of original acquisition in nm/px.
    x_col, y_col, sigma_col : str
        Column names in df.

    Returns
    -------
    df_out : pandas DataFrame
        Copy of df with added columns:
        x_px, y_px, sigma_px, mean_intensity
    """
    # Open image
    image = Image.open(path_image).convert("RGB")

    # Convert to grayscale
    gray = rgb2gray(image)

    # Current image size
    jpg_w, jpg_h = image.size
    orig_w, orig_h = orig_size

    # Scaling factors
    sx = jpg_w / orig_w
    sy = jpg_h / orig_h

    H, W = gray.shape

    # Storage lists
    x_list = []
    y_list = []
    sigma_list = []
    mean_list = []

    for _, row in df.iterrows():

        x_nm = row[x_col]
        y_nm = row[y_col]
        sigma_nm = row[sigma_col]

        # nm → original pixels
        x_orig_px = x_nm / pixel_size_original_nm
        y_orig_px = y_nm / pixel_size_original_nm
        sigma_orig_px = sigma_nm / pixel_size_original_nm

        # original pixels → current image pixels
        x_px = int(round(x_orig_px * sx))
        y_px = int(round(y_orig_px * sy))
        sigma_px = int(round(sigma_orig_px * sx))

        # Build circular mask (clipped automatically)
        rr, cc = disk((y_px, x_px), sigma_px, shape=(H, W))
        mask = np.zeros((H, W), dtype=bool)
        mask[rr, cc] = True

        # Compute mean intensity
        if mask.sum() > 0:
            mean_intensity = gray[mask].mean()
        else:
            mean_intensity = np.nan

        x_list.append(x_px)
        y_list.append(y_px)
        sigma_list.append(sigma_px)
        mean_list.append(mean_intensity)

    # Return modified copy
    df_out = df.copy()
    df_out["x_px"] = x_list
    df_out["y_px"] = y_list
    df_out["sigma_px"] = sigma_list
    df_out["mean_intensity"] = mean_list

    return df_out

def main(path1, path2):
    path1 = str(path1).strip()
    path2 = str(path2).strip()

    # Make final table
    results = aggregate_data(path1, path2)

    # Spearman correlation
    corr = sprearman_correlation(results)

    # Results export
    results.to_csv(f"{path2}/results.csv", index=False)
    corr.to_csv(f"{path2}/spearman_pairs.csv", index=False)

    print(f"Saved: {path2}/'results.csv'")
    print(f"Saved: {path2}/'spearman_pairs.csv'")

 

if __name__ == "__main__":
    path1 =  "/mnt/c/Users/Elena/Desktop/Data_processing/020226_U2OS_fixed_WT" # path to the folder containing the csv files with nucleus area and MFI
    path2 = "/mnt/c/Users/Elena/Desktop/Data_processing/020226_U2OS_fixed_WT/res2_new" # path to the folder containing the csv files with foci number and MFI
    
    main(path1, path2)