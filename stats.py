from pathlib import Path
import re
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage.draw import disk
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
#from scipy.stats import spearmanr


def key_from_csv(p: Path) -> str:
    """
    C2...nd2_(series_01)_0233-0247.csv  ->  C2...nd2_(series_01)
    """
    name = p.stem  # no .csv
    # remove trailing _####-#### (or similar) if present
    name = re.sub(r"_\d+-\d+$", "", name)
    name = re.sub(r"_roi$", "", name, flags=re.IGNORECASE)
    return name

def key_from_img(p: Path) -> str:
    """
    C2...nd2_(series_01).jpg -> C2...nd2_(series_01)
    """
    return p.stem  # no .extention

def MFI_foci(
        image_path,
        df,
        px_size_ts_x = 11.6,
        px_size_ts_y = 11.6,
        px_size_x = 57.5,
        px_size_y = 58.7,
        x_col="x [nm]",
        y_col="y [nm]",
        sigma_col="sigma [nm]"
    ):

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Convert image to grayscale
        gray = rgb2gray(image)
        H, W = gray.shape # number of pixels

        # Scaling factors
        sx = px_size_ts_x/px_size_x
        sy = px_size_ts_y/px_size_y
        ssigma = np.mean([px_size_ts_x, px_size_ts_y]) / np.mean([px_size_x, px_size_y])

        # Storage lists
        x_list = []
        y_list = []
        sigma_list = []
        mean_list = []

        for _, row in df.iterrows():
            x_nm = row[x_col]
            y_nm = row[y_col]
            sigma_nm = row[sigma_col]

            # original pixels → current image pixels
            x_px = int(round(sx * x_nm / px_size_ts_x))
            y_px = int(round(sy * y_nm / px_size_ts_y))
            sigma_px = max(1, int(round(ssigma * sigma_nm / np.mean([px_size_ts_x, px_size_ts_y])))) # minimal possible value is 1 pixel!

            # Build circular mask (clipped automatically)
            rr, cc = disk((y_px, x_px), sigma_px, shape=(H, W))
            mask = np.zeros((H, W), dtype=bool)
            mask[rr, cc] = True
            disk((y_px, x_px), sigma_px, shape=(H, W))

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


def aggregate_nuclei_data(dir_nuclei_stat):
    # Paths to files
    nuclei_path = Path(str(dir_nuclei_stat).strip()) # path to data about nucleus in total

    # Check path
    if not nuclei_path.exists():
        raise FileNotFoundError(f"dir1 not found: {nuclei_path}")
    
    # Check that there are .csv files
    nuclei_files = sorted(nuclei_path.glob("*.csv"))
    if not nuclei_files:
        raise FileNotFoundError(f"No CSV files found in: {nuclei_path}")
    
    dfs = []

    for f in nuclei_files:
        key = key_from_csv(f)
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
        dfs.append(df)

    final = pd.concat(dfs, ignore_index=True)

    return final

def plot_histogram(df, column, bins=50,
                   xlabel=None,
                   title=None,
                   figsize=(4, 3),
                   dpi=300,
                   save_path=None):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.hist(
        df[column].dropna(),
        bins=bins,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=12)

    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # --- Save if path provided ---
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)


def MFI_foci_all(dir_images, dir_foci):
    # Paths to files
    images_path = Path(str(dir_images).strip())
    foci_data_path = Path(str(dir_foci).strip())

    # Check path
    if not images_path.exists():
        raise FileNotFoundError(f"Directory {images_path} is not found!")
    if not foci_data_path.exists():
        raise FileNotFoundError(f"Directory {foci_data_path} is not found!")
    
    # Check that there are files
    images = sorted(images_path.glob("*.tif"))
    if not images:
        raise FileNotFoundError(f"No .TIF files found in: {images_path}")
    foci = sorted(
        f for f in foci_data_path.glob("*.csv")
        if not f.stem.endswith(("_roi", "_extent"))
    )
    if not foci:
         raise FileNotFoundError(f"No .CSV files found in: {foci_data_path}")
    
    # --- Make list of tuples called pairs = [(image_path, csv foci filem path)] ---
    img_by_key = {key_from_img(p): p for p in images} # dictionary {image name: image path}
    pairs = []
    for f in foci:
        k = key_from_csv(f)
        img_path = img_by_key.get(k)
        pairs.append((f, img_path))
    print(f"Found {len(pairs)} (image.tif foci.csv) pairs.")
        
    # Calculate MFI of each foci
    for file, image in pairs:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df_added = MFI_foci(image_path = image,
                            df = df,
                            px_size_ts_x = 11.6,
                            px_size_ts_y = 11.6,
                            px_size_x = 57.5,
                            px_size_y = 58.7,
                            x_col="x [nm]",
                            y_col="y [nm]",
                            sigma_col="sigma [nm]"
                            )
        
        # Filtration based on sigma_nm value
        filtered = df_added[df_added["sigma [nm]"] > 75]
        
        # Calculate outliers based on mean intensity of foci
        data = filtered["mean_intensity"]
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        # Create new bool column 'Outlier'
        filtered["Outlier"] = filtered["mean_intensity"] > upper_bound
        n_outliers = sum(filtered["Outlier"])

        print(f"File {key_from_csv(file)}: keep {filtered.shape[0]} out of {df_added.shape[0]} foci. Number of outliers: {n_outliers}")
        
        # Export
        new_name = key_from_csv(file) + "_extent.csv"
        new_path = file.with_name(new_name)
        filtered.to_csv(new_path, index=False) # export new extended dataframe

        #print(f"File {new_name} is saved.")
    
def aggregation_foci(dir):
    path_files = Path(str(dir).strip())
    files = sorted(path_files.glob("*_extent.csv"))
    foci_rows = []

    # generic function
    check_column_mean = lambda df, col: (
            float(df[col].mean())
            if col in df.columns and not df.empty
            else pd.NA
    )

    for f in files:
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
            "Outliers_sigma_nm": check_column_mean(df[df["Outlier"] == True], "sigma [nm]")
        })

    foci_summary = pd.DataFrame(foci_rows)

    return foci_summary

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


def main(p1, p2, output_dir):
    df_nuclei = aggregate_nuclei_data(dir_nuclei_stat = p1)
    MFI_foci_all(dir_images = p1, dir_foci = p2)
    results = aggregation_foci(dir = p2)

    merged = df_nuclei.merge(results, on="File_name", how="left")

    # Results export
    merged.to_csv(f"{output_dir}/results.csv", index=False)
    print(f"Aggregated results are saved: to the directory: {output_dir}.")
 
if __name__ == "__main__":
    p1 = "./examples" # path to directory with nucleus Area and Mean
    p2 = "./examples/run" # path to ThunderSTORM data
    output_dir = "./examples"
    
    main(p1, p2, output_dir)