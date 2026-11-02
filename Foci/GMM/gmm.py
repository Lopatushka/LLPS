import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import ndimage as ndi

from sklearn.mixture import GaussianMixture
from skimage import io, filters, morphology, measure
from skimage.color import rgb2gray

# =====================================================
# 1. LOAD IMAGE
# =====================================================
# =====================================================
# 2. LOAD IMAGE
# =====================================================

def load_image(image_path):

    image = io.imread(image_path)

    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = rgb2gray(image[..., :3])

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image, got {image.shape}"
        )

    image = image.astype(np.float64)

    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or Inf")

    imin = image.min()
    imax = image.max()

    if imax <= imin:
        raise ValueError("Image has constant intensity")

    return (image - imin) / (imax - imin)


# =====================================================
# 3. FIT GMM
# =====================================================

def fit_gmm(image):

    pixels = image.reshape(-1, 1)

    if (
        max_fit_pixels is not None
        and len(pixels) > max_fit_pixels
    ):
        rng = np.random.default_rng(random_state)

        indices = rng.choice(
            len(pixels),
            size=max_fit_pixels,
            replace=False
        )

        fit_pixels = pixels[indices]

    else:
        fit_pixels = pixels

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        reg_covar=1e-6
    )

    gmm.fit(fit_pixels)

    return gmm


# =====================================================
# 4. GMM PARAMETERS AND THRESHOLD
# =====================================================

def get_gmm_parameters(gmm):

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.reshape(-1))
    weights = gmm.weights_

    order = np.argsort(means)

    return (
        means[order],
        stds[order],
        weights[order]
    )


def find_gmm_threshold(gmm, means, grid_size=10000):

    intensity_grid = np.linspace(0, 1, grid_size)

    probabilities = gmm.predict_proba(
        intensity_grid.reshape(-1, 1)
    )

    brightest_component = np.argmax(
        gmm.means_.flatten()
    )

    predicted_components = np.argmax(
        probabilities,
        axis=1
    )

    candidates = np.where(
        (intensity_grid >= means[-2])
        & (predicted_components == brightest_component)
    )[0]

    if len(candidates) == 0:
        raise ValueError(
            "No high-intensity GMM decision region found"
        )

    return intensity_grid[candidates[0]]


# =====================================================
# 5. INITIAL SEGMENTATION
# =====================================================

def segment_image(smooth, threshold):

    mask = smooth > threshold

    mask = ndi.binary_fill_holes(mask)

    mask = morphology.remove_small_objects(
        mask,
        min_size=min_area,
        connectivity=connectivity
    )

    labels = measure.label(
        mask,
        connectivity=connectivity
    )

    return mask, labels


# =====================================================
# 6. LOCAL ROI REFINEMENT
# =====================================================

def refine_gmm_rois(
    smooth,
    labels,
    alpha=0.35,
    min_area=1,
    background_percentile=10
):

    refined_labels = np.zeros(
        labels.shape,
        dtype=np.int32
    )

    next_label = 1

    for region in measure.regionprops(labels):

        minr, minc, maxr, maxc = region.bbox

        local_image = smooth[
            minr:maxr,
            minc:maxc
        ]

        local_roi = (
            labels[minr:maxr, minc:maxc]
            == region.label
        )

        intensities = local_image[local_roi]

        background = np.percentile(
            intensities,
            background_percentile
        )

        peak = intensities.max()

        local_threshold = (
            background
            + alpha * (peak - background)
        )

        refined = (
            local_roi
            & (local_image >= local_threshold)
        )

        components = measure.label(
            refined,
            connectivity=connectivity
        )

        peak_position = np.unravel_index(
            np.argmax(
                np.where(
                    local_roi,
                    local_image,
                    -np.inf
                )
            ),
            local_image.shape
        )

        peak_label = components[peak_position]

        if peak_label == 0:
            continue

        refined = components == peak_label

        if np.count_nonzero(refined) < min_area:
            continue

        local_output = refined_labels[
            minr:maxr,
            minc:maxc
        ]

        local_output[refined] = next_label

        next_label += 1

    refined_mask = refined_labels > 0

    return refined_mask, refined_labels


# =====================================================
# 7. DRAW CONTOURS
# =====================================================

def draw_contours(ax, image, labels):

    ax.imshow(
        image,
        cmap="gray",
        interpolation="nearest"
    )

    for region in measure.regionprops(labels):

        roi_mask = labels == region.label

        contours = measure.find_contours(
            np.pad(
                roi_mask.astype(float),
                1,
                mode="constant"
            ),
            level=0.5
        )

        for contour in contours:

            ax.plot(
                contour[:, 1] - 1,
                contour[:, 0] - 1,
                color="red",
                linewidth=0.7
            )

    ax.axis("off")




def main():
    # =====================================================
    # PARAMETERS AND FOLDERS
    # =====================================================

    input_dir = Path("images")
    output_dir = Path("GMM_refined_segmentation")

    figures_dir = output_dir / "figures"
    masks_dir = output_dir / "masks"
    labels_dir = output_dir / "labels"
    measurements_dir = output_dir / "measurements"
    
    for folder in [
    output_dir,
    figures_dir,
    masks_dir,
    labels_dir,
    measurements_dir
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    # Load all TIFF files
    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
    ]) if input_dir.exists() else []

    if not image_paths:
        raise FileNotFoundError(
            f"No TIFF images found in {input_dir.resolve()}"
        )

    print(f"Found {len(image_paths)} images")
    
    # Segmentation parameters
    sigma = 1.2

    n_components = 3
    random_state = 42
    n_init = 10
    max_iter = 300
    max_fit_pixels = None

    min_area = 3
    connectivity = 2

    alpha = 0.35
    background_percentile = 10
    refined_min_area = 1

    hist_bins = 150