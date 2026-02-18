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


def key_from_csv(p: Path) -> str:
    """
    C2...nd2_(series_01)_0233-0247.csv  ->  C2...nd2_(series_01)
    """
    name = p.stem  # no .csv
    # remove trailing _####-#### (or similar) if present
    name = re.sub(r"_\d+-\d+$", "", name)
    return name

def key_from_img(p: Path) -> str:
    """
    C2...nd2_(series_01).jpg -> C2...nd2_(series_01)
    """
    return p.stem  # no .jpg

def compute_mean_intensity_from_localizations(
        image_path,
        df,
        orig_size=(2560, 2560),
        pixel_size_original_nm=16.0,
        x_col="x [nm]",
        y_col="y [nm]",
        sigma_col="sigma [nm]"
    ):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Convert image to grayscale
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
        #sigma_px = int(round(sigma_orig_px * sx))
        sigma_px = max(1, int(round(sigma_orig_px * sx))) # minimal possible value is 1 pixel!

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

        key = key_from_csv(f)
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

    # --- build image lookup by key ---
    img_by_key = {key_from_img(p): p for p in images_paths}

    # --- make pairs (csv, image) ---
    pairs = []
    missing_images = []

    # --- Create .csv - image pairs ---
    files2 = sorted(dir_path2.glob("*.csv"))
    if not files2:
        raise FileNotFoundError(f"No CSV files found in dir2: {dir_path2}")
    
    # --- Create .csv - image pairs ---
    for f in files2:
        k = key_from_csv(f)
        img_path = img_by_key.get(k)
        if img_path is None:
            missing_images.append(f)
            continue
        pairs.append((f, img_path))
    
    print(f"Pairs found: {len(pairs)}")

    # Calculate MFI of each foci
    for file, image in pairs:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df_added = compute_mean_intensity_from_localizations(image_path = image,
                                                    df = df,
                                                    orig_size=(2560, 2560),
                                                    pixel_size_original_nm=16.0,
                                                    x_col="x [nm]",
                                                    y_col="y [nm]",
                                                    sigma_col="sigma [nm]"
                                                 )
        # Filtration based on sigma_nm value
        filtered = df_added[df_added["sigma [nm]"] > 75]
        print(f"Filtration of file {key_from_csv(file)}: keep {df_added.shape[0]} out of {filtered.shape[0]} foci")

        # Outliers
        data = filtered["sigma [nm]"]
        #mean = data.mean()
        #std = data.std()
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        # Create new column bool
        filtered["Outlier"] = filtered["mean_intensity"] > upper_bound
        
        new_name = key_from_csv(file) + "_extent.csv"
        new_path = file.with_name(new_name)
        filtered.to_csv(new_path, index=False) # export new extended dataframe
    
    # ---- FOCI SUMMARY (path2) ----
    files3 = sorted(dir_path2.glob("*_extent.csv"))
    foci_rows = []

    # generic function
    check_column_mean = lambda df, col: (
        float(df[col].mean())
        if col in df.columns and not df.empty
        else pd.NA
    )

    for f in files3:
        k = key_from_csv(f)
        k = k[:-7]
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()

        # Count rows
        foci_rows.append({
        "File_name": k,
        "Foci_number": int(df.shape[0]),
        "All_foci_IFI_photons": check_column_mean(df, "intensity [photon]"),
        "All_foci_MFI_px": check_column_mean(df, "mean_intensity"),
        "All_foci_sigma_nm": check_column_mean(df, "sigma [nm]"),
        "Outliers_number": sum(df["Outlier"]),
        "Outliers_MFI_px": check_column_mean(df[df["Outlier"] == True], "mean_intensity"),
        "Outliers_sigma_nm": check_column_mean(df[df["Outlier"] == True], "sigma_nm")
        })

    foci_summary = pd.DataFrame(foci_rows)

    # ---- MERGE ----
    final["File_name"] = final["File_name"].astype(str).str.strip()
    foci_summary["File_name"] = foci_summary["File_name"].astype(str).str.strip()

    merged  = final.merge(foci_summary, on="File_name", how="left")

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
    p1 = "/mnt/c/Users/Elena/Desktop/Data_processing/test" # path to directory with original images and nucleus Area and Mean
    p2 = "/mnt/c/Users/Elena/Desktop/Data_processing/test/res" # path to ThunderSTORM data
    
    main(p1, p2)