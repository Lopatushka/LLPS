import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.draw import disk
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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

def draw_foci(image_path, df, showplot = True, save_image = True, save_path = ""):
    # Load image
    image = Image.open(image_path)
    #image_name = filename(image_path)
    arr = np.array(image) # convert image to numpy matrix

    # Plot image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr, cmap="gray")

    # Draw red circles
    for _, row in df.iterrows():
        circle = Circle(
            (row["x_pixel"], row["y_pixel"]),
            row["sigma_pixel"],
            fill=False,
            edgecolor="red",
            linewidth=1
        )

        ax.add_patch(circle)

    # Match image coordinates
    ax.set_xlim(0, arr.shape[1])
    ax.set_ylim(arr.shape[0], 0)

    # Show image
    if showplot:
        plt.show(fig)

    # Save image
    if save_image: 
        plt.savefig(save_path,
            dpi=300,
            bbox_inches="tight"
        )

        # Do not display image
        plt.close(fig)

def foci_one_image(image_path, df, px_size_nm, plot = True, show_plot = True, save_image = True, save_path = ""):
    image = Image.open(image_path)  # load image
    image_name = filename(image_path) # get image name
    arr = np.array(image) # convert image to numpy matrix
    H, W = arr.shape # get number of pixels (512*512 for 16-bit image)

    # Storage lists
    x_list = []
    y_list = []
    sigma_list = []
    mean_list = []

    # Prepare dataframe
    df.columns = df.columns.str.strip()  # remove hidden spaces in headers
    df = df.rename(columns={"x [nm]": "x_nm", "y [nm]": "y_nm", "sigma [nm]": "sigma_nm", "intensity [photon]": "intensity_photon"}) # Rename columns

    # Iteration through the thunderSTORM dataframe
    for _, row in df.iterrows():
        x_px = int(row["x_nm"] / px_size_nm)
        y_px = int(row["y_nm"] / px_size_nm)
        r_px = max(1, int(row["sigma_nm"] / px_size_nm)) # minimal possible value for radius is 1 pixel!

        # Build circular mask (clipped automatically)
        rr, cc = disk((y_px, x_px), r_px, shape=(H, W))
        mask = np.zeros((H, W), dtype=bool)
        mask[rr, cc] = True

        n_pixels_mask = np.sum(mask)

        # Compute mean intensity
        if n_pixels_mask > 0:
            mean_intensity = arr[mask].mean()
        else:
            mean_intensity = np.nan

        # Add values to the corresponding lists
        x_list.append(x_px)
        y_list.append(y_px)
        sigma_list.append(r_px)
        mean_list.append(mean_intensity)
    
    # Return modified copy
    df_out = df.copy()
    df_out["x_pixel"] = x_list
    df_out["y_pixel"] = y_list
    df_out["sigma_pixel"] = sigma_list
    df_out["foci_MFI"] = mean_list

    # Make a plot
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(arr, cmap="gray")

        for _, row in df_out.iterrows():
            x = row["x_pixel"]
            y = row["y_pixel"]
            r = row["sigma_pixel"]

            circle = Circle(
            (x, y),
            r,
            fill=False,
            edgecolor="red",
            linewidth=1
            )

            ax.add_patch(circle)
        
        # Match image coordinates
        ax.set_xlim(0, arr.shape[1])
        ax.set_ylim(arr.shape[0], 0)

        # Show plot
        if show_plot:
            plt.show(fig)

        # Save image
        if save_image:
            plt.savefig(save_path,
                dpi=300,
                bbox_inches="tight"
            )

        # Do not display image
        plt.close(fig)

    return df_out

def plot_histogram(df, column, bins=50,
                   xlabel=None,
                   title=None,
                   figsize=(4, 3),
                   dpi=300,
                   save_image = True,
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
    #plt.show()

    # --- Save if path provided ---
    if save_image:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)


def main():
    # Ask about paths with data and output directory to save results
    # Example of path: /mnt/c/users/elopatukhin/Desktop/Miscroscopy/160226_U2OS_fixed/MP_WT_0.3

    #nuclei_dir = check_directory(input("Enter pathway to the directory with the information about nuclei (Area and Mean): "))
    dir_images = check_directory(input("Enter pathway to the directory with the images: "))
    dir_foci = check_directory(input("Enter pathway to the directory with the information about foci (ThunderSTORM output): "))
    px = float(input("Enter the pixel size in nm [default value is 58.739]: ") or 58.739)

    while True:
        answer = input("Save results in the same folder as foci? (Y/N): ").strip().upper()
        if answer == "Y":
            output_dir = dir_foci
            #output_dir = nuclei_dir # temporaly!!!
            break
        elif answer == "N":
            output_dir = check_directory(input("Enter output folder path: ").strip())
            break
        else:
            print("Please enter Y or N.")

    # --- Processed nuclei info (Area, Mean) ---
    #nuclei_info = nuclei_data(nuclei_dir, output_dir) # export results

    # --- Process foci data ---
    # List of paths to the images
    paths_images = [
    os.path.join(dir_images, f)
    for f in os.listdir(dir_images)
    if os.path.isfile(os.path.join(dir_images, f))
    and f.lower().endswith(".tif") and "_ROI_".lower() in f.lower()
    ]

    # List of paths to the foci.csv
    paths_foci_csv = [
        os.path.join(dir_foci, f)
        for f in os.listdir(dir_foci)
        if os.path.isfile(os.path.join(dir_foci, f))
        and f.lower().endswith(".csv")   
        ]

    # Create dictionaries
    img_by_key = {filename(image_name): image_name for image_name in paths_images} # dictionary {image name w/o ext: image path}
    csv_by_key = {filename(csv_name)[:-5]: csv_name for csv_name in paths_foci_csv} # dictionary {csv file name w/o ext: image path}
    combined = {k: (img_by_key[k], csv_by_key[k]) for k in img_by_key} # dictionary {file_name: (path_to_image, path_to_foci_csv)}

    n_images = len(combined)
    print(f"Founded {n_images} pairs of image.tif : foci.csv files.")

    # Iteration through combined dictioanry
    for name, (img_path, csv_path) in combined.items():
        df = pd.read_csv(csv_path)
        
        # Foci analysis and plot
        result = foci_one_image(img_path,
                                df,
                                px_size_nm = px,
                                plot = True,
                                show_plot = False,
                                save_image = True,
                                save_path = f"{output_dir}/{name}_foci_map.png")
        
        # Save results
        path_to_result = os.path.join(output_dir, f"{name}_foci_processed.csv")
        result.to_csv(path_to_result, index=False)
        #all_foci.append(result)
        
        # Make a threshold for Sigma_nm
        Q1 = np.percentile(result["sigma_nm"], 25)
        Q3 = np.percentile(result["sigma_nm"], 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR

        # Plot histogram for sigma_nm with upper bound and save the plot
        path_to_hist = os.path.join(output_dir, f"{name}_sigma_hist.png")
        plot_histogram(df = result, column = "sigma_nm", bins=50,
                   xlabel= "Sigma, nm",
                   title = name,
                   figsize=(4, 3),
                   dpi=300,
                   save_image = True,
                   save_path=path_to_hist,
                   threshold = upper_bound)

        # Make filtration for Sigma_nm
        result_filetered = result[result["sigma_nm"] <= upper_bound]

        # Save filtered results
        path_to_result_filtered = os.path.join(output_dir, f"{name}_foci_processed_filtered.csv")
        result_filetered.to_csv(path_to_result_filtered, index=False)

        # Plot foci from filtered dataframe
        save_foci_filtered = os.path.join(output_dir, f"{name}_foci_map_filtered.png")
        draw_foci(image_path = img_path,
                  df = result_filetered,
                  showplot = False,
                  save_image = True,
                  save_path = save_foci_filtered)

        print(f"Sucessfully processed image {name}.")

 
if __name__ == "__main__":
    main()