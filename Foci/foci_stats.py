import os
#from pathlib import Path
#import re
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

def draw_foci_with_radius(image_path, df, px_size_um, output_path):







def foci_one_image(image_path, df, px_size_um, output_path):
    image = Image.open(image_path)  # load image
    image_name = filename(image_path) # get image name
    arr = np.array(image) # convert image to numpy matrix
    H, W = arr.shape # get number of pixels (512*512 for 16-bit image)

    # Iteration through the thunderSTORM dataframe
    for _, row in df.iterrows():
        x_px = int(row["x_um"] / px_size_um)
        y_px = int(row["y_um"] / px_size_um)
        r_px = math.ceil(row["sigma_um"] / px_size_um)

        # Build circular mask (clipped automatically)
        rr, cc = disk((y_px, x_px), r_px, shape=(H, W))
        mask = np.zeros((H, W), dtype=bool)
        mask[rr, cc] = True

        # Compute mean intensity
        if mask.sum() > 0:
            mean_intensity = arr[mask].mean()
        else:
            mean_intensity = np.nan

    # Plot image
    #fig, ax = plt.subplots(figsize=(8, 8))
    #ax.imshow(arr, cmap="gray")

    # Draw red circles
    for _, row in df.iterrows():
        x_px = row["x_um"] / px_size_um
        y_px = row["y_um"] / px_size_um
        r_px = row["sigma_um"] / px_size_um

        circle = Circle(
            (x_px, y_px),
            r_px,
            fill=False,
            edgecolor="red",
            linewidth=1
        )

        ax.add_patch(circle)

    # Match image coordinates
    ax.set_xlim(0, arr.shape[1])
    ax.set_ylim(arr.shape[0], 0)

    # Save imafe
    plt.savefig(
        f"{output_path}/{image_name}_foci_sigma.png",
        dpi=300,
        bbox_inches="tight"
    )

    # Do not display image
    plt.close(fig)




 

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

def nuclei_data(dir, output_dir):
    paths_csv_files = [
    os.path.join(dir, f)
    for f in os.listdir(dir)
    if os.path.isfile(os.path.join(dir, f))
    and f.lower().endswith(".csv")
    ]

    # Number of .csv files and check
    n = len(paths_csv_files)
    if n == 0:
        raise FileNotFoundError(f"No CSV files found in the directory: {dir}")
    print(f"Founded {n} .csv files")

    # Create list of dataframes
    dfs = []
    for f in paths_csv_files:
        filname = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()  # remove hidden spaces in headers

        # Ensure expected columns exist
        if "Area" not in df.columns or "Mean" not in df.columns:
            raise KeyError(
                f"In nuclei file {f.name} expected columns 'Area' and 'Mean'. "
                f"Found: {list(df.columns)}"
            )
        
        df["Filename"] = filname # Add Filename column
        df = df.rename(columns={"Area": "Nucleus_area", "Mean": "Nucleus_MFI"}) # Rename columns
        df = df[["Filename", "Nucleus_area", "Nucleus_MFI"]] # change columns order
        dfs.append(df) # add df to the list of dataframes

    # Make final dataframe and export
    final = pd.concat(dfs, ignore_index=True)
    final.to_csv(f"{output_dir}/nuclei_results.csv", index=False)

    return final

def plot_histogram(df, column, bins=50,
                   xlabel=None,
                   title=None,
                   figsize=(4, 3),
                   dpi=300,
                   save_path=None,
                   threshold = 0):

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

    ax.axvline(threshold, linestyle="--", linewidth=2)

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

        # Plot histogram of foci mean and intensity and save it
        plot_path = file.with_name(key_from_csv(file) + "_hist.jpg")
        plot_histogram(df = filtered, column = "mean_intensity", bins=50,
                   xlabel="Foci mean intensity",
                   title=key_from_csv(file),
                   figsize=(4, 3),
                   dpi=300,
                   save_path=plot_path,
                   threshold = upper_bound)

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


def _main1(p1, p2, output_dir):
    df_nuclei = aggregate_nuclei_data(dir_nuclei_stat = p1)
    MFI_foci_all(dir_images = p1, dir_foci = p2)
    results = aggregation_foci(dir = p2)

    merged = df_nuclei.merge(results, on="File_name", how="left")

    # Results export
    merged.to_csv(f"{output_dir}/results.csv", index=False)
    print(f"Aggregated results.csv file is saved in the directory: {output_dir}.")

def main():
    # Ask about paths with data and output directory to save results
    # Example of path: /mnt/c/users/elopatukhin/Desktop/Miscroscopy/160226_U2OS_fixed/MP_WT_0.3
    nuclei_dir = check_directory(input("Enter pathway to the directory with the information about nuclei (Area and Mean): "))
    #foci_dir = check_directory(input("Enter pathway to the directory with the information about foci (ThunderSTORM output): "))
    while True:
        answer = input("Save results in the same folder as foci? (Y/N): ").strip().upper()
        if answer == "Y":
            #output_dir = foci_dir
            output_dir = nuclei_dir # temporaly!!!
            break
        elif answer == "N":
            #output_dir = check_directory(input("Enter output folder path: ").strip())
            break
        else:
            print("Please enter Y or N.")

    # --- Processed nuclei info (Area, Mean) ---
    nuclei_info = nuclei_data(nuclei_dir, output_dir) # export results


 
if __name__ == "__main__":
    main()


    
    