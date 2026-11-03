"""Compare G4 versus G3+G4 from previously saved smoothed TIFF images.

Install: pip install numpy pandas scipy scikit-image scikit-learn matplotlib tifffile
Run: python compare_G4_vs_G3G4.py

GMM is fitted ONCE per image, using the saved smoothed intensities.
Component numbers are ordered by increasing fitted mean intensity.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Save without opening/blocking Linux/WSL windows
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from scipy.stats import norm
from skimage import io, measure, morphology
from sklearn.mixture import GaussianMixture

N_COMPONENTS = 4
PROBABILITY_THRESHOLD = 0.5
MIN_AREA = 3
FILL_HOLES = True
N_INIT = 5
RANDOM_STATE = 42
MAX_FIT_PIXELS = 30000
FIG_DPI = 180
CHUNK_SIZE = 250000

def ask_directory(message):
    return Path(input(message).strip().strip('"').strip("'")).expanduser()


def make_folders(root):
    folders = {}
    for comparison in ('G4', 'G3_G4', 'difference'):
        folders[comparison] = {}
        for sub in ('binary_masks', 'labeled_rois', 'roi_measurements', 'panels'):
            p = root / comparison / sub
            p.mkdir(parents=True, exist_ok=True)
            folders[comparison][sub] = p
    for sub in ('histograms', 'summary', 'models'):
        p = root / sub
        p.mkdir(parents=True, exist_ok=True)
        folders[sub] = p
    return folders

def get_images(previous_root):
    # Accept either the previous results root or its smoothed_images folder.
    candidate = previous_root / 'smoothed_images'
    source = candidate if candidate.is_dir() else previous_root
    images = sorted(p for p in source.iterdir()
                    if p.is_file() and p.suffix.lower() in ('.tif', '.tiff'))
    if not images:
        raise FileNotFoundError(f'No smoothed TIFFs found in {source}')
    return images

def fit_gmm(image):
    pixels = image.reshape(-1, 1)
    if MAX_FIT_PIXELS is not None and len(pixels) > MAX_FIT_PIXELS:
        rng = np.random.default_rng(RANDOM_STATE)
        fit_pixels = pixels[rng.choice(len(pixels), MAX_FIT_PIXELS, replace=False)]
    else:
        fit_pixels = pixels
    gmm = GaussianMixture(n_components=N_COMPONENTS, n_init=N_INIT,
                          covariance_type='full', reg_covar=1e-6,
                          random_state=RANDOM_STATE)
    gmm.fit(fit_pixels)
    return gmm


def make_mask(prob, min_area=MIN_AREA):
    mask = prob >= PROBABILITY_THRESHOLD
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    if FILL_HOLES:
        mask = binary_fill_holes(mask)
    return np.asarray(mask, dtype=bool)

def measurements(mask, intensity, name):
    labels = measure.label(mask, connectivity=2)
    props = measure.regionprops_table(
        labels, intensity_image=intensity,
        properties=('label', 'area', 'centroid', 'equivalent_diameter_area',
                    'perimeter', 'eccentricity', 'solidity', 'mean_intensity',
                    'max_intensity', 'min_intensity')
    )
    df = pd.DataFrame(props)
    df.insert(0, 'image', name)
    return labels, df

def save_mask_and_rois(mask, intensity, name, folder, key):
    labels, df = measurements(mask, intensity, name)
    io.imsave(folder['binary_masks'] / f'{name}_{key}_mask.tif',
              (mask.astype(np.uint8) * 255), check_contrast=False)
    io.imsave(folder['labeled_rois'] / f'{name}_{key}_labels.tif',
              labels.astype(np.int32), check_contrast=False)
    df.to_csv(folder['roi_measurements'] / f'{name}_{key}_ROIs.csv', index=False)
    return labels, df

def save_panel(image, mask, name, title, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Smoothed input')
    axs[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title(f'{title} (area={int(mask.sum()):,} px)')
    for ax in axs:
        ax.axis('off')
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)

def save_comparison(image, mask4, mask34, name, path):
    added = mask34 & ~mask4
    lost = mask4 & ~mask34
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    for ax, data, title in zip(
        axs,
        (image, mask4, mask34, added),
        ('Smoothed input', 'G4', 'G3 + G4', 'Added pixels (G3+G4 minus G4)')
    ):
        ax.imshow(data, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(f'{name} | added={added.sum():,} px; lost={lost.sum():,} px')
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    
def save_histogram(image, gmm, order, name, path):
    values = image.ravel()
    hi = float(np.percentile(values, 99.99))
    lo = float(values.min())
    if hi <= lo:
        hi = float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, 201)
    counts, _ = np.histogram(values, bins=edges)
    density = counts / (len(values) * np.diff(edges))
    x = np.linspace(lo, hi, 1200)
    means = gmm.means_.ravel()
    stds = np.sqrt(gmm.covariances_.reshape(-1))
    weights = gmm.weights_
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for ax, log in zip(axs, (False, True)):
        ax.stairs(density, edges, fill=True, alpha=.25, color='gray', label='Pixel density')
        total = np.zeros_like(x)
        for rank, idx in enumerate(order, start=1):
            pdf = weights[idx] * norm.pdf(x, means[idx], stds[idx])
            total += pdf
            ax.plot(x, pdf, lw=2 if rank in (3, 4) else 1.2,
                    label=f'G{rank}: mean={means[idx]:.2f}, weight={weights[idx]:.3f}')
        ax.plot(x, total, color='black', lw=1.8, label='Total GMM')
        if log:
            ax.set_yscale('log')
        ax.set_title('Log density' if log else 'Linear density')
        ax.set_xlabel('Smoothed intensity')
        ax.set_ylabel('Probability density')
        ax.legend(fontsize=7)
    fig.suptitle(f'{name}: GMM K=4 (components ordered by mean)')
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    

def count_roi_overlap(labels_a, labels_b):
    """Counts ROI splitting/merging based on pixel overlaps (not one-to-one matching)."""
    overlap = (labels_a > 0) & (labels_b > 0)
    pairs = np.unique(np.column_stack((labels_a[overlap], labels_b[overlap])), axis=0)
    a_ids = set(map(int, labels_a[labels_a > 0].ravel()))
    b_ids = set(map(int, labels_b[labels_b > 0].ravel()))
    linked_a = set(map(int, pairs[:, 0])) if len(pairs) else set()
    linked_b = set(map(int, pairs[:, 1])) if len(pairs) else set()
    from collections import Counter
    a_per_b = Counter(int(b) for a, b in pairs)
    b_per_a = Counter(int(a) for a, b in pairs)
    return {
        'g4_rois_without_overlap': len(a_ids - linked_a),
        'g3g4_rois_without_overlap': len(b_ids - linked_b),
        'g3g4_rois_merging_multiple_g4_rois': sum(n > 1 for n in a_per_b.values()),
        'g4_rois_splitting_into_multiple_g3g4_rois': sum(n > 1 for n in b_per_a.values()),
    }


def main():
    previous_root = ask_directory('Previous results folder (containing smoothed_images/): ')
    output_root = ask_directory('New comparison output folder: ')
    images = get_images(previous_root)
    folders = make_folders(output_root)
    summary_rows = []
    roi_tables = {'G4': [], 'G3_G4': [], 'difference': []}
    for i, path in enumerate(images, start=1):
        name = path.stem.removesuffix('_smoothed')
        print(f'[{i}/{len(images)}] {path.name}: loading', flush=True)
        image = io.imread(path)
        if image.ndim != 2:
            raise ValueError(f'{path}: expected 2D TIFF, got shape {image.shape}')
        image = image.astype(np.float64)
        if not np.isfinite(image).all():
            raise ValueError(f'{path}: non-finite intensity values')
        print('  Fitting GMM once...', flush=True)
        gmm = fit_gmm(image)
        order = np.argsort(gmm.means_.ravel())
        # Store model for exact reproducibility, without refitting next time.
        import pickle
        with open(folders['models'] / f'{name}_gmm.pkl', 'wb') as f:
            pickle.dump(gmm, f)
        # A compact model parameters table is also saved.
        pd.DataFrame({
            'component': np.arange(1, 5),
            'mean': gmm.means_.ravel()[order],
            'std': np.sqrt(gmm.covariances_.reshape(-1))[order],
            'weight': gmm.weights_[order],
        }).to_csv(folders['models'] / f'{name}_components.csv', index=False)
        p4 = np.empty(image.size, dtype=np.float32)
        p34 = np.empty(image.size, dtype=np.float32)
        flat = image.reshape(-1, 1)
        for start in range(0, image.size, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, image.size)
            posterior = gmm.predict_proba(flat[start:stop])
            p4[start:stop] = posterior[:, order[3]]
            p34[start:stop] = posterior[:, order[2]] + posterior[:, order[3]]
        # Each mask is postprocessed independently, exactly as in the prior pipeline.
        mask4 = make_mask(p4.reshape(image.shape))
        mask34 = make_mask(p34.reshape(image.shape))
        added = mask34 & ~mask4
        lost = mask4 & ~mask34
        labels4, rois4 = save_mask_and_rois(mask4, image, name, folders['G4'], 'G4')
        labels34, rois34 = save_mask_and_rois(mask34, image, name, folders['G3_G4'], 'G3_G4')
        _, rois_added = save_mask_and_rois(added, image, name, folders['difference'], 'added')
        for key, df in (('G4', rois4), ('G3_G4', rois34), ('difference', rois_added)):
            roi_tables[key].append(df)
        save_panel(image, mask4, name, 'G4', folders['G4']['panels'] / f'{name}_G4.png')
        save_panel(image, mask34, name, 'G3 + G4', folders['G3_G4']['panels'] / f'{name}_G3_G4.png')
        save_comparison(image, mask4, mask34, name,
                        folders['difference']['panels'] / f'{name}_comparison.png')
        save_histogram(image, gmm, order, name, folders['histograms'] / f'{name}_histograms.png')
        a4, a34 = int(mask4.sum()), int(mask34.sum())
        summary_rows.append({
            'image': name,
            'G4_area_px': a4,
            'G3_G4_area_px': a34,
            'area_difference_px': a34 - a4,
            'area_increase_pct_vs_G4': 100 * (a34 - a4) / a4 if a4 else np.nan,
            'added_pixels': int(added.sum()),
            'lost_pixels': int(lost.sum()),
            'G4_n_rois': len(rois4),
            'G3_G4_n_rois': len(rois34),
            'roi_count_difference': len(rois34) - len(rois4),
            'added_region_count': len(rois_added),
            'intersection_px': int((mask4 & mask34).sum()),
            'union_px': int((mask4 | mask34).sum()),
            'iou': float((mask4 & mask34).sum() / (mask4 | mask34).sum()) if (mask4 | mask34).any() else np.nan,
            **count_roi_overlap(labels4, labels34),
        })
        print(f'  G4: {a4:,} px, {len(rois4)} ROIs | G3+G4: {a34:,} px, {len(rois34)} ROIs | +{added.sum():,} pixels', flush=True)
    pd.DataFrame(summary_rows).to_csv(folders['summary'] / 'comparison_summary.csv', index=False)
    for key, dfs in roi_tables.items():
        pd.concat(dfs, ignore_index=True).to_csv(
            folders['summary'] / f'all_ROIs_{key}.csv', index=False)
    print(f'\nDone. Results saved to: {output_root.resolve()}', flush=True)


if __name__ == '__main__':
    main()
