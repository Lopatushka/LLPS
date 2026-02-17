# Import necessary libraries
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage.draw import disk
from matplotlib.patches import Circle

def scatter_plot(df, x_col, y_col, 
                 x_lim=None, 
                 y_lim=None):

    plt.figure(figsize=(4, 4), dpi=150)
    plt.scatter(df[x_col], df[y_col], alpha=0.6)

    plt.xlabel(x_col.replace("_", " "))
    plt.ylabel(y_col.replace("_", " "))

    # Fixed scale
    if x_lim is not None:
        plt.xlim(x_lim)

    if y_lim is not None:
        plt.ylim(y_lim)

    plt.tight_layout()
    plt.show()



def plot_histogram(df, column, bins=50, xlabel=None, title=None,
                   figsize=(4, 3), dpi=200):
    """
    Plot histogram for a dataframe column.

    Parameters:
    df : pandas.DataFrame
    column : str
        Column name to plot
    bins : int
        Number of bins
    xlabel : str (optional)
    title : str (optional)
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.hist(
        df[column].dropna(),
        bins=bins,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8
    )

    ax.set_xlabel(xlabel if xlabel else column, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title if title else f"Distribution of {column}", fontsize=12)

    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_foci_on_image(
    gray_image,
    df,
    x_col="x_px",
    y_col="y_px",
    r_col="sigma_px",
    circle_color="red",
    center_size=6,
    linewidth=1,
    figsize=(7, 7),
    show=True
):
    """
    Plot detected foci as circles on a grayscale image.

    Parameters
    ----------
    gray_image : 2D numpy array
        Grayscale image.
    df : pandas DataFrame
        DataFrame containing coordinates and radii.
    x_col, y_col : str
        Column names for x and y coordinates.
    r_col : str
        Column name for radius (in pixels).
    circle_color : str
        Color of circle and center dot.
    center_size : int
        Size of center dot.
    linewidth : int
        Circle outline thickness.
    figsize : tuple
        Figure size.
    show : bool
        Whether to call plt.show().
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(gray_image, cmap="gray")

    for x_px, y_px, r_px in zip(df[x_col], df[y_col], df[r_col]):
        # circle outline
        ax.add_patch(
            Circle(
                (x_px, y_px),
                r_px,
                fill=False,
                edgecolor=circle_color,
                linewidth=linewidth
            )
        )
        # center dot
        ax.scatter(x_px, y_px, c=circle_color, s=center_size)

    ax.axis("off")

    if show:
        plt.show()

    return fig, ax

def main(dir_path):
    files = sorted(dir_path.glob("*_extent.csv"))
    print(files)

if __name__ == "__main__":
    print("kkk")
    dir = "./examples"
    main(dir)