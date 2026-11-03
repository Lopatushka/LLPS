"""ORC1 four-image Standard GMM segmentation.
Install: pip install numpy pandas matplotlib scipy scikit-image scikit-learn tifffile
Put exactly four 2D TIFF files in ./images and run this script.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_fill_holes
from scipy.stats import norm
from skimage import io, measure, morphology, restoration
from sklearn.mixture import GaussianMixture
import time

# ========== CONFIG ==========
#IMAGE_FOLDER = Path('images')
#OUTPUT_FOLDER = Path('gmm_results')

N_COMPONENTS = 4
SIGNAL_COMPONENTS = 1
PROBABILITY_THRESHOLD = 0.5
ROLLING_BALL_RADIUS = 50
SMOOTH_SIGMA = 1.0
MIN_AREA = 3
FILL_HOLES = True
N_INIT = 5
RANDOM_STATE = 42
HIST_BINS = 200
HIST_PERCENTILE = 99.99
FIG_DPI = 180
MAX_FIT_PIXELS = 30000 # For large images, set to e.g. 150000 to reduce GMM fit time.

def _log_step(message, start):
    print(
        f"{message}: {time.perf_counter() - start:.2f} s",
        flush=True
    )
    return time.perf_counter()


def validate_parameters():
    if not (1 <= SIGNAL_COMPONENTS <= N_COMPONENTS):
        raise ValueError('SIGNAL_COMPONENTS must be between 1 and N_COMPONENTS')
    if not (0 < PROBABILITY_THRESHOLD <= 1):
        raise ValueError('PROBABILITY_THRESHOLD must be in (0, 1]')
    if ROLLING_BALL_RADIUS <= 0 or SMOOTH_SIGMA < 0 or MIN_AREA < 1:
        raise ValueError('Check rolling ball radius, sigma, and minimum area')


def process_image(path):
    t = time.perf_counter()
     
    original = io.imread(path)
    if original.ndim != 2:
        raise ValueError(f'{path.name}: expected 2D grayscale TIFF, got {original.shape}')
    
    original = original.astype(np.float64)
    if not np.isfinite(original).all():
        raise ValueError(f'{path.name}: image contains NaN or infinity')
    
    #t = log_step("Image loaded", t)
    
    background = restoration.rolling_ball(original, radius=ROLLING_BALL_RADIUS)
    #t = log_step("Rolling ball completed", t)
     
    corrected = np.maximum(original - background, 0)
    smoothed = gaussian_filter(corrected, sigma=SMOOTH_SIGMA)
    
    #t = log_step("Smoothing completed", t)
    
    pixels = smoothed.reshape(-1, 1)
    
    if MAX_FIT_PIXELS is not None and len(pixels) > MAX_FIT_PIXELS:
        rng = np.random.default_rng(RANDOM_STATE)
        fit_pixels = pixels[rng.choice(len(pixels), MAX_FIT_PIXELS, replace=False)]
    else:
        fit_pixels = pixels
        
    if len(fit_pixels) < N_COMPONENTS:
        raise ValueError('Not enough pixels for the requested GMM components')
    
    gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type='full',
                          n_init=N_INIT, random_state=RANDOM_STATE, reg_covar=1e-6)
    gmm.fit(fit_pixels)
    
    #t = log_step("GMM fitting completed", t)
     
    order = np.argsort(gmm.means_.ravel())
    signal_indices = order[-SIGNAL_COMPONENTS:]
    
    # Chunk posterior calculation to avoid large allocations for big images.
    signal_prob = np.empty(len(pixels), dtype=np.float32)
    for start in range(0, len(pixels), 250000):
        stop = min(start + 250000, len(pixels))
        signal_prob[start:stop] = gmm.predict_proba(pixels[start:stop])[:, signal_indices].sum(axis=1)
    
    signal_prob = signal_prob.reshape(original.shape)
    
    #t = log_step("Probabilities completed", t)
    
    mask = signal_prob >= PROBABILITY_THRESHOLD
    mask = morphology.remove_small_objects(mask, min_size=MIN_AREA)
    
    if FILL_HOLES:
        mask = binary_fill_holes(mask)
        
    labels = measure.label(mask, connectivity=2)
    
    #t = log_step("Segmentation completed", t)
    
    props = measure.regionprops_table(labels, intensity_image=original,
        properties=('label', 'area', 'centroid', 'equivalent_diameter_area',
                    'perimeter', 'eccentricity', 'solidity', 'mean_intensity', 'max_intensity', 'min_intensity'))
    
    rois = pd.DataFrame(props)
    rois.insert(0, 'image', path.name)
    
    #t = log_step("ROI measurements completed", t)
    
    return dict(original=original, background=background, corrected=corrected,
                smoothed=smoothed, mask=mask, labels=labels, gmm=gmm,
                signal_indices=signal_indices, signal_prob=signal_prob, rois=rois)


def histogram_data(image, gmm, bins=HIST_BINS, percentile=HIST_PERCENTILE):
    values = image.ravel()
    lo = float(values.min())
    hi = float(np.percentile(values, percentile))
    if hi <= lo:
        hi = float(values.max())
    if hi <= lo:
        hi = lo + 1
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    density = counts / (len(values) * np.diff(edges))
    x = np.linspace(lo, hi, 1500)
    means = gmm.means_.ravel()
    stds = np.sqrt(gmm.covariances_.reshape(-1))
    weights = gmm.weights_
    order = np.argsort(means)
    pdfs = {int(k): weights[k] * norm.pdf(x, means[k], stds[k]) for k in order}
    total = sum(pdfs.values())
    return edges, density, x, order, pdfs, total, means, stds, weights

def _draw_histogram_ax(ax, hist, signal_indices, log=False, legend=True):
    edges, density, x, order, pdfs, total, means, stds, weights = hist

    ax.stairs(
        density, edges,
        fill=True,
        color="gray",
        alpha=0.32,
        label="Pixel intensities"
    )

    colors = plt.get_cmap("tab10")
    signal_set = set(map(int, signal_indices))

    for rank, k in enumerate(order):
        selected = int(k) in signal_set

        ax.plot(
            x,
            pdfs[int(k)],
            color=colors(rank % 10),
            linewidth=2.2 if selected else 1.3,
            linestyle="-" if selected else "--",
            label=(
                f"G{rank+1}: μ={means[k]:.2f}, "
                f"σ={stds[k]:.2f}, w={weights[k]:.3f}"
                + (" [SIGNAL]" if selected else "")
            )
        )

    ax.plot(x, total, color="black", lw=2.3, label="Total GMM")

    if log:
        ax.set_yscale("log")
        positives = np.r_[density[density > 0], total[total > 0]]

        if len(positives):
            ax.set_ylim(
                bottom=max(float(np.min(positives)) * 0.5, 1e-12)
            )

    ax.set_title(
        "Histogram + GMM (" + ("log Y" if log else "linear Y") + ")"
    )
    ax.set_xlabel("Smoothed, background-corrected intensity")
    ax.set_ylabel("Probability density")
    ax.grid(alpha=0.2)

    if legend:
        ax.legend(fontsize=7)


def draw_histogram(
    result,
    name=None,
    save_path=None,
    figsize=(16, 5),
    show=False
):
    """
    Create linear and logarithmic GMM histograms.

    If save_path is provided, save the figure.
    If show=True, display the figure.
    Always close the figure afterward.
    """
    hist = histogram_data(
        result["smoothed"],
        result["gmm"]
    )

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    _draw_histogram_ax(
        axs[0],
        hist,
        result["signal_indices"],
        log=False
    )

    _draw_histogram_ax(
        axs[1],
        hist,
        result["signal_indices"],
        log=True
    )

    if name is not None:
        fig.suptitle(f"{name}: GMM intensity distributions", fontsize=14)

    fig.tight_layout()

    try:
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_path,
                dpi=FIG_DPI,
                bbox_inches="tight"
            )

        if show:
            plt.show()
    finally:
        plt.close(fig)

def plot_results(result, name, output=None):
    """
    Create a six-panel visualization of GMM segmentation.

    If output is provided, save the figure.
    The figure is automatically closed after saving/displaying.
    """

    original = result["original"]
    background = result["background"]
    corrected = result["corrected"]
    smoothed = result["smoothed"]
    mask = result["mask"]
    labels = result["labels"]

    hist = histogram_data(smoothed, result["gmm"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # First four panels
    images = [
        original,
        background,
        corrected,
        mask
    ]

    titles = [
        "Original image",
        f"Rolling-ball background (r={ROLLING_BALL_RADIUS})",
        "Background corrected (before smoothing)",
        "Binary mask (filled holes)"
    ]

    for ax, img, title in zip(axes.flat[:4], images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    # Fifth panel: Numbered ROIs
    ax = axes[1, 1]
    ax.imshow(original, cmap="gray")

    if mask.any() and (~mask).any():
        ax.contour(
            mask.astype(float),
            levels=[0.5],
            colors="lime",
            linewidths=0.7
        )

    for region in measure.regionprops(labels):
        y, x = region.centroid

        ax.text(
            x, y,
            str(region.label),
            color="yellow",
            fontsize=6,
            ha="center",
            va="center"
        )

    ax.set_title(f"Numbered ROIs (N={labels.max()})")
    ax.axis("off")

    # Sixth panel: GMM histogram
    _draw_histogram_ax(
        axes[1, 2],
        hist,
        result["signal_indices"],
        log=True
    )

    # Figure title
    fig.suptitle(
        f"{name} | K={N_COMPONENTS}, "
        f"signal K={SIGNAL_COMPONENTS}, "
        f"σ={SMOOTH_SIGMA}",
        fontsize=15
    )

    fig.tight_layout()

    # Optional saving
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            output / f"{name}_six_panels.png",
            dpi=FIG_DPI,
            bbox_inches="tight"
        )

    plt.close(fig)


def main():
    validate_parameters()
    
    # Ask user for directories
    input_dir = input(
        "Enter the input directory containing TIFF images: "
    ).strip().strip('"').strip("'")
    
    output_dir = input(
    "Enter the output directory for GMM results: "
    ).strip().strip('"').strip("'")
    
    IMAGE_FOLDER = Path(input_dir).expanduser()
    OUTPUT_FOLDER = Path(output_dir).expanduser()
    
    # Validate input directory
    if not IMAGE_FOLDER.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {IMAGE_FOLDER}"
        )
        
    # Create output directory if necessary
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )
    
    # Create output subfolders
    OUTPUT_DIRS = {
        "histograms": OUTPUT_FOLDER / "histograms",
        "panels": OUTPUT_FOLDER / "panels",
        "backgrounds": OUTPUT_FOLDER / "backgrounds",
        "corrected": OUTPUT_FOLDER / "corrected_images",
        "smoothed": OUTPUT_FOLDER / "smoothed_images",
        "masks": OUTPUT_FOLDER / "binary_masks",
        "labels": OUTPUT_FOLDER / "labeled_rois",
        "measurements": OUTPUT_FOLDER / "roi_measurements",
        "summary": OUTPUT_FOLDER / "summary",
    }
    
    for folder in OUTPUT_DIRS.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\nInput directory: {IMAGE_FOLDER.resolve()}")
    print(f"Output directory: {OUTPUT_FOLDER.resolve()}\n")
    
    # Check images in the input directory
    paths = sorted(p for p in IMAGE_FOLDER.iterdir() if p.is_file() and p.suffix.lower() in ('.tif', '.tiff'))
    if len(paths) == 0:
        raise ValueError(f'No TIFF images found in {IMAGE_FOLDER.resolve()}')
    
    all_roi_tables, summary = [], []
    
    # Import images and process them
    for path in paths:
        
        print(f"\nProcessing: {path.name}", flush=True)
        
        results = process_image(path)
        name = path.stem
        
        # Six-panel visualization
        plot_results(
            results,
            name=name,
            output=OUTPUT_DIRS["panels"]
        )
        
        # Separate GMM histogram
        # This assumes your histogram function accepts save_path.
        draw_histogram(
            results,
            name=name,
            save_path=OUTPUT_DIRS["histograms"] / f"{name}_histograms.png"
        )

        # Background image
        io.imsave(
            OUTPUT_DIRS["backgrounds"] / f"{name}_background.tif",
            results["background"].astype(np.float32),
            check_contrast=False
        )

        # Background-corrected image
        io.imsave(
            OUTPUT_DIRS["corrected"] / f"{name}_corrected.tif",
            results["corrected"].astype(np.float32),
            check_contrast=False
        )

        # Smoothed image
        io.imsave(
            OUTPUT_DIRS["smoothed"] / f"{name}_smoothed.tif",
            results["smoothed"].astype(np.float32),
            check_contrast=False
        )

        # Binary mask
        io.imsave(
            OUTPUT_DIRS["masks"] / f"{name}_mask.tif",
            results["mask"].astype(np.uint8) * 255,
            check_contrast=False
        )

        # Labeled ROIs
        io.imsave(
            OUTPUT_DIRS["labels"] / f"{name}_labels.tif",
            results["labels"].astype(np.int32),
            check_contrast=False
        )

        # ROI measurements
        roi_df = results["rois"].copy()

        roi_df.to_csv(
            OUTPUT_DIRS["measurements"] / f"{name}_ROIs.csv",
            index=False
        )

        all_roi_tables.append(roi_df)

        summary.append({
            "image": path.name,
            "n_components": N_COMPONENTS,
            "signal_components": SIGNAL_COMPONENTS,
            "n_rois": int(results["labels"].max()),
            "total_area_px": int(results["mask"].sum()),
            "area_fraction": float(results["mask"].mean())
        })

    # Save combined results
    combined_df = pd.concat(all_roi_tables, ignore_index=True)

    combined_df.to_csv(
        OUTPUT_DIRS["summary"] / "all_ROIs.csv",
        index=False
    )

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        OUTPUT_DIRS["summary"] / "summary.csv",
        index=False
    )

    print("\nAll images processed successfully.")
       

if __name__ == '__main__':
    main()