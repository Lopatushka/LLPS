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


def foci_one_image(image_path, df, px_size_nm, output_path, plot = True, show_plot = True, save_image = True):
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
        #print(x_px, x_px, r_px, n_pixels_mask, mean_intensity)

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
            plt.savefig(
                f"{output_path}/{image_name}_sigma.png",
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