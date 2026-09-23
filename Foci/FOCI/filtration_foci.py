import pandas as pd
import numpy as np
import os
from pathlib import Path
from PIL import Image
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

# For Ayriscan detector filration conditions are the following:
#
# Sigma, nm - [100, 1000]
# MFI > 100
# MFI/SD of FI > 5
# If to circles are overllaped > 30%, keep only the biggest

def circles_overlap(c1, c2):
    x1, y1, r1 = c1
    x2, y2, r2 = c2

    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    return distance < r1 + r2

def overlap_fraction(c1, c2):
    """
    c1, c2 = (x, y, radius)

    Возвращает долю площади МЕНЬШЕГО круга,
    которая перекрывается другим кругом.
    """

    x1, y1, r1 = c1
    x2, y2, r2 = c2

    # Расстояние между центрами
    d = np.hypot(x1 - x2, y1 - y2)

    # 1. Круги вообще не пересекаются
    if d >= r1 + r2:
        return 0.0

    # Радиус меньшего круга
    r_small = min(r1, r2)

    # 2. Меньший круг полностью находится внутри большего
    if d <= abs(r1 - r2):
        return 1.0

    # 3. Частичное пересечение
    alpha = np.arccos(
        (d**2 + r1**2 - r2**2) / (2 * d * r1)
    )

    beta = np.arccos(
        (d**2 + r2**2 - r1**2) / (2 * d * r2)
    )

    intersection_area = (
        r1**2 * alpha
        + r2**2 * beta
        - 0.5 * np.sqrt(
            (-d + r1 + r2)
            * (d + r1 - r2)
            * (d - r1 + r2)
            * (d + r1 + r2)
        )
    )

    small_area = np.pi * r_small**2

    return intersection_area / small_area

def remove_overlapping_circles(
    df,
    x_col="X",
    y_col="Y",
    radius_col="Radius",
    threshold=0.8
):
    # Work on a copy
    df = df.copy()

    # Check larger circles first
    sorted_indices = df[radius_col].sort_values(ascending=False).index

    selected_indices = []

    for idx in sorted_indices:

        circle = (
            df.loc[idx, x_col],
            df.loc[idx, y_col],
            df.loc[idx, radius_col]
        )

        remove = False

        for selected_idx in selected_indices:

            bigger_circle = (
                df.loc[selected_idx, x_col],
                df.loc[selected_idx, y_col],
                df.loc[selected_idx, radius_col]
            )

            overlap = overlap_fraction(circle, bigger_circle)

            if overlap > threshold:
                remove = True
                break

        if not remove:
            selected_indices.append(idx)

    # Keep only circles that survived
    return df.loc[selected_indices].sort_index()


def plot_histogram(df, column, bins=50,
                   xlabel=None,
                   title=None,
                   figsize=(4, 3),
                   dpi=300,
                   save_image = True,
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

    #ax.axvline(threshold, linestyle="--", linewidth=2)

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
    
def draw_foci(image_path, df, showplot = True, save_image = True, save_path = ""):
    # Load image
    image = Image.open(image_path)
    arr = np.array(image) # convert image to numpy matrix

    # Plot image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr, cmap="gray")

    # Draw red circles
    for _, row in df.iterrows():
        x = row["x_pixel"]
        y = row["y_pixel"]
        r = row["sigma_pixel"]

        circle = Circle(
            (x, y),
            r,
            fill=False,
            edgecolor="red",
            linewidth=0.25
        )

        ax.add_patch(circle)

    # Match image coordinates
    ax.set_xlim(0, arr.shape[1])
    ax.set_ylim(arr.shape[0], 0)

    # Save image
    if save_image: 
        plt.savefig(save_path,
            dpi=300,
            bbox_inches="tight"
        )
        
        # Do not display image
        #plt.close(fig)

    # Show image
    if showplot:
        plt.show(fig)
    
def df_filtration(path_to_df, hist = True, plot = True, path_to_img = ""):
    # Process the filename from .csv path
    df_path = Path(path_to_df)
    file_name = df_path.stem
    
    if path_to_img != "":
        img_path = Path(path_to_img)
        img_name = img_path.stem
    
    # Load dataframe
    df = pd.read_csv(df_path)
        
    # --- Perform filrations ---
    # Sigma filtation
    df = df[(df['sigma_nm'] > 100) & (df['sigma_nm'] < 1000)]
    
    # MFI filtration
    df = df[df['foci_MFI'] > 100]
    
    # S.d. filtration
    df[df['foci_MFI'] / df['foci_SD'] > 8]
    
    # Overlapping
    df = remove_overlapping_circles(
        df,
        x_col="x_pixel",
        y_col="y_pixel",
        radius_col="sigma_pixel",
        threshold=0.3
    )
    
    if hist:
        # Plot and save histograms of original images
        plot_histogram(df, column = "sigma_nm", bins=50,
                    xlabel="Sigma, nm",
                    title=file_name,
                    figsize=(4, 3),
                    dpi=300,
                    save_image = True,
                    save_path = df_path.with_name(file_name + "_sigma_hist.png")
                    )
    
        plot_histogram(df, column = "foci_MFI", bins=50,
                    xlabel="MFI of foci",
                    title=file_name,
                    figsize=(4, 3),
                    dpi=300,
                    save_image = True,
                    save_path = df_path.with_name(file_name + "_foci_MFI_hist.png")
                    )
    
        plot_histogram(df, column = "foci_SD", bins=50,
                    xlabel="S.d. of fluorescent intencity of foci",
                    title=file_name,
                    figsize=(4, 3),
                    dpi=300,
                    save_image = True,
                    save_path = df_path.with_name(file_name + "_foci_sd_hist.png")
                    )
        if plot:
            draw_foci(image_path = img_path,
            df = df,
            showplot = False,
            save_image = True,
            save_path = img_path.with_name(img_name + "_filtred_mapped.png"))

# -----------------
# MAIN FUNCTION
# -----------------
def main():
    pass

if __name__ == "__main__":
    main()