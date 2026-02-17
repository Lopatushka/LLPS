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

    # Read dataframe with single nucleus data
    #df = pd.read_csv(df_path, encoding="latin1")

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
    #print(img_by_key)

    # --- make pairs (csv, image) ---
    pairs = []
    missing_images = []

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
    
    # Create .csv - image pairs 
    for f in files2:
        k = key_from_csv(f)
        img_path = img_by_key.get(k)
        if img_path is None:
            missing_images.append(f)
            continue
        pairs.append((f, img_path))
    
    print(f"Pairs found: {len(pairs)}")
    print(f"CSV without matching image: {len(missing_images)}")

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
        new_name = key_from_csv(file) + "_extent.csv"
        new_path = file.with_name(new_name)
        df_added.to_csv(new_path, index=False)
        

p1 = "/mnt/c/Users/Elena/Desktop/Data_processing/sb" 
p2 = "/mnt/c/Users/Elena/Desktop/Data_processing/sb/res" 
aggregate_data(p1, p2)