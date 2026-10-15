import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from skimage import io, filters, morphology, measure, segmentation
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries
from scipy import ndimage as ndi

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
    
)

from skimage import segmentation, exposure

def plot_image(image, clean_mask, labels, df_subset,
               mode="image", figsize=(8, 8)):

    # Get original segmentation labels from DataFrame subset
    selected_ids = df_subset["label"].astype(int).to_numpy()

    # Keep only selected objects
    selected_labels = np.where(
        np.isin(labels, selected_ids),
        labels,
        0
    )

    # Choose background
    if mode == "image":
        background = exposure.rescale_intensity(
            image.astype(float),
            in_range="image",
            out_range=(0, 1)
        )

    elif mode == "mask":
        background = clean_mask.astype(float)

    else:
        raise ValueError("mode must be 'image' or 'mask'")

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(
        segmentation.mark_boundaries(
            background,
            selected_labels,
            color=(1, 0, 0)
        )
    )

    ax.set_title(
        f"Selected objects: {len(selected_ids)}"
    )

    ax.axis("off")

    plt.tight_layout()
    plt.show()
    
def show_objects_from_list(image, labels, df, indexes, padding=10, ncols=3):

    if np.isscalar(indexes):
        indexes = [indexes]

    indexes = list(indexes)

    regions = {
        r.label: r
        for r in measure.regionprops(labels)
    }

    if not indexes:
        print("No objects selected.")
        return

    nrows = int(np.ceil(len(indexes) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.5, nrows * 3.5),
        squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    for i, idx in enumerate(indexes):

        row = df.loc[idx]

        # Segmentation label
        object_id = int(row["label"])

        region = regions.get(object_id)

        if region is None:
            print(f"Object {object_id} not found.")
            continue

        min_row, min_col, max_row, max_col = region.bbox

        y0 = max(0, min_row - padding)
        y1 = min(image.shape[0], max_row + padding)

        x0 = max(0, min_col - padding)
        x1 = min(image.shape[1], max_col + padding)

        crop = image[y0:y1, x0:x1]
        mask_crop = labels[y0:y1, x0:x1] == object_id

        r = i // ncols
        c = i % ncols

        ax = axes[r, c]

        ax.imshow(crop, cmap="gray")

        ax.contour(
            mask_crop.astype(float),
            levels=[0.5],
            colors="red",
            linewidths=1
        )

        # Original index from full DataFrame
        ax.set_title(
            f"Label: {idx}\n"
            f"Area: {row['area']:.2f} | "
            f"Circularity: {row['circularity']:.2f}",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()
    
def objects_show(subset, image, labels,
                 crop_size=40, ncols=5):

    n = len(subset)

    if n == 0:
        print("No objects to display.")
        return

    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.5, nrows * 2.5),
        squeeze=False
    )

    H, W = image.shape[:2]

    for ax, (_, row) in zip(axes.flat, subset.iterrows()):

        x = int(round(row["centroid-1"]))
        y = int(round(row["centroid-0"]))

        # Original watershed object ID
        object_id = int(row["label"])

        half = crop_size // 2

        x1 = max(0, x - half)
        x2 = min(W, x + half)
        y1 = max(0, y - half)
        y2 = min(H, y + half)

        # Crop original image
        crop = image[y1:y2, x1:x2]

        # Crop watershed labels
        labels_crop = labels[y1:y2, x1:x2]

        # Select only this object
        mask = labels_crop == object_id

        # Find original watershed boundary
        boundary = find_boundaries(mask, mode="inner")

        # Show microscopy image
        ax.imshow(crop, cmap="gray")

        # Overlay boundary in red
        ax.contour(
            mask.astype(float),
            levels=[0.5],
            colors="red",
            linewidths=1
        )

        ax.set_title(
            f"Index: {row.name}\n"
            f"Circularity: {row['circularity']:.2f}\n"
            f"Area: {row['area']:.2f}",
            fontsize=9
        )

        ax.axis("off")

    # Hide unused panels
    for ax in axes.flat[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()