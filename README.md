# Foci — nuclear foci segmentation and analysis

Tools for segmenting nuclei and nuclear foci in fluorescence microscopy images,
measuring foci morphology and intensity, and comparing experimental conditions
with multivariate statistics (PCA, UMAP, PERMANOVA).

> Status: work in progress

## Workflow

```
raw .tif
   │  FIJI/nuclei_manual_segment.py   (Fiji)  → one .tif per nucleus + ROI .zip
   ▼
single-nucleus .tif images
   │  FIJI/substract_bg.py   (Fiji, optional) → *_bg_subtracted.tif
   │  FOCI/mfi_nuclei.py                         → MFI_nuclei.csv (optional)
   │  FOCI/outsu.py  or  FOCI/gmm.py             → per-image foci tables
   │  FOCI/agg.py   (both methods)               → aggregated_data.csv (one row per cell)
   ▼
aggregated_data.csv for each condition
   │  STATISTICS/METRICS.ipynb, CSAT.ipynb, PCA_v2.ipynb, UMAP_v2.ipynb
   ▼
plots and statistical tests
```

## Folder structure

```
LLPS/
├── FIJI/                  Jython scripts, run inside Fiji/ImageJ
├── FOCI/                  Python scripts for foci segmentation and measurement
│   ├── outsu.py           Otsu threshold foci segmentation   (config_outsu.toml)
│   ├── gmm.py             Gaussian-mixture-model foci segmentation (config_gmm.toml)
│   ├── config_outsu.toml  parameters of outsu.py
│   ├── config_gmm.toml    parameters of gmm.py
│   ├── agg.py             aggregates the foci tables of either method, one row per cell
│   └── mfi_nuclei.py      nuclear area and mean intensity (MFI) of every image (optional)
├── STATISTICS/            PCA / UMAP / METRICS / CSAT analysis notebooks and helpers
│
└── requirements.txt       Python dependencies
```

### `FIJI/` — Fiji/ImageJ scripts

Run these from Fiji's Script Editor (language: Python/Jython). They work on the
images currently open in Fiji and ask for an output folder.

| Script | What it does |
|---|---|
| `nuclei_manual_segment.py` | For each open image you draw nuclear ROIs on the DAPI channel and add them to the ROI Manager. Each nucleus is saved as a separate `.tif` of the measurement channel (`C<channel>_<image>_ROI_<nn>_<roi>.tif`), with every pixel outside the ROI set to 0 on all slices, and the ROIs as `<image>_rois.zip`. The ROI dialog has three buttons: **OK** saves and continues, **Skip image** closes the image and its split channels without saving and goes to the next one, **Cancel** stops the batch. Images with the same output name in one batch get `_2`, `_3`, ... instead of overwriting each other. Split-channel windows (`C1-...`) are not processed. |
| `substract_bg.py` | *Optional.* Rolling-ball background subtraction (radius in px, asked in a dialog) on every slice of every open image; saves `<image>_bg_subtracted.tif` and writes the radius and the progress to the Log. Images already named `*_bg_subtracted` are skipped. Run it on the single-nucleus images from `nuclei_manual_segment.py`: for these masked images the zero area around the nucleus is first filled with the nearest edge values, so the rolling ball does not create a bright rim at the nuclear edge, and the outside is set back to 0 afterwards. Unmasked images are processed as usual. |
| `opener.py` | Opens, in alphabetical order, all files in a folder whose name contains a given pattern (case-insensitive; empty = any) and has a given extension. Optionally also searches subfolders. Unreadable files are skipped and listed in the Log. |
| `closer.py` | Closes all open images after one confirmation dialog, discarding unsaved changes (no "Save?" prompt per image). Tables and the Log stay open. |

If you use `substract_bg.py`, keep `rolling_ball_radius = 0` in `config_outsu.toml` /
`config_gmm.toml`, otherwise the background is subtracted twice. Note that
`partition_coefficient`, `fraction_signal_in_foci` and the nucleus/nucleoplasm
intensities are meant to be measured on raw intensities; on background-subtracted
input they describe the subtracted image instead.

### `FOCI/outsu.py` — Otsu threshold segmentation

Both `outsu.py` and `gmm.py` expect single-nucleus images with everything outside
the nucleus set to 0: the non-zero pixels are the nucleus. Foci are only searched
inside it, and `area_fraction` in `summary.csv` is foci area / nucleus area.

- **`outsu.py`**.

**Input:**  every `.tif`/`.tiff` files in the input folder.

**Pipeline:**

1. *optional* rolling-ball background subtraction (`rolling_ball_radius`, `0` = off, the
  default)
2. Gaussian smoothing (`sigma`)
3. Otsu threshold (computed on the nucleus, i.e. the non-zero pixels) × `threshold_factor`
4. small-object removal (`min_area`)
5. foci ROIs: `segmentation = "watershed"` (default; distance transform + watershed to split touching foci, seeds `min_distance` apart;
  fragments smaller than `min_area` are merged into the touching focus; a focus
  split only into such fragments is kept unsplit) or
  `"connected"` (connected regions of the mask); ROIs smaller than `min_area` are then removed.
  
  Parameters are set in `config_outsu.toml`.
  
  **Outputs** go into the same subfolders as `gmm.py`:
  1. `roi_measurements/` (`<image>_ROIs.csv`,
  one row per focus, see *Measured features* below),
  2. `summary/`
  (`all_ROIs.csv`, and `summary.csv` with the Otsu threshold, the threshold used,
  number of foci, nucleus area and foci area per image)
  3. `binary_masks/`
  4. `labeled_rois/`
  5. `backgrounds/`
  6. `corrected_images/`
  7. `smoothed_images/`
  8.  `histograms/` (`<image>_histograms.png`: intensity histogram with the threshold).
  With `rolling_ball_radius > 0` the intensities are measured on the
  background-subtracted image, otherwise on the raw image.
  9. `panels/` if `save_figures = true` (`<image>_six_panels.png`: original,
  background, corrected, mask, distance transform, segmented ROIs) and
 
### `FOCI/gmm.py` — Gaussian mixture model segmentation

**`gmm.py`** — fits a GMM (`n_components`, 4 by default) to pixel intensities 
of each image; pixels whose probability of belonging to the brightest
component(s) (`signal_components`) is at least `probability_threshold` form
the foci mask. The mask is split into ROIs by `segmentation`:
 - `"watershed"` (default; distance transform + watershed, seeds `min_distance` apart;
fragments smaller than `min_area` are merged into the touching focus; a focus
  split only into such fragments is kept unsplit)
- `"connected"` (connected regions); ROIs smaller than `min_area` are then removed.

*Optional* rolling-ball background subtraction (`rolling_ball_radius`, `0` =
off) and Gaussian smoothing. With `rolling_ball_radius > 0` the foci
intensities (`mean/max/min_intensity`) are measured on the
background-subtracted image, otherwise on the raw image.

Parameters are set in `config_gmm.toml`.

**Outputs** are saved into subfolders:
1. `histograms/`
2. `panels/`
3. `backgrounds/`
4. `corrected_images/`
5. `smoothed_images/`
6. `binary_masks/`
7. `labeled_rois/`
8. `roi_measurements/` (`*_ROIs.csv`)
9. `summary/`

### `FOCI/agg.py` — aggregation (both methods)

Reads all `*_ROIs.csv` files in the `roi_measurements/` folder of an `outsu.py`
or `gmm.py` run and writes `aggregated_data.csv` with one row per cell: `n_foci`
plus the median of each foci feature, and the per-cell metrics from `summary/summary.csv` of the same run:
- `foci_density`
- `area_fraction`
- `nucleus_mean_intensity`
- `nucleoplasm_mean_intensity`
- `fraction_signal_in_foci`
- `size_gini`
- `clark_evans_ratio`
- `fraction_touching`

Cells with zero foci are kept (`n_foci = 0`).

### *Optional* `FOCI/mfi_nuclei.py` — nuclear area and mean intensity

For every
`.tif`/`.tiff` in the input folder it measures the nucleus (pixels > 0) and writes `MFI_nuclei.csv` with the following columns:
- `image_id`
- `Filename`
- `Area` (calibrated from the
TIFF resolution tags, e.g. micron^2)
- `unit`
- `Area_px`
- `Mean`

`Area_px` and `Mean` equal `nucleus_area_px` and `nucleus_mean_intensity` in `summary.csv` of
`outsu.py` / `gmm.py` for the same images. An existing `MFI_nuclei.csv` is never
overwritten (`MFI_nuclei_1.csv`, `_2`, ...).

### `STATISTICS/` — condition comparison

## **`METRICS.ipynb`** + `METRICS_utils.py` - **per-metric comparison of conditions**:
- SuperPlots (cells coloured by replicate, replicate
  medians)
- Welch t-test on replicate medians (cell-level Mann–Whitney as
  descriptive)
- median ratio with hierarchical-bootstrap 95 % CI
- Cliff's delta
- Benjamini–Hochberg FDR over all metrics × comparisons
- effect-size heatmap
- expression-level check (Spearman with `nucleus_mean_intensity`, expression-matched re-analysis).
  
The `data` list has one entry per condition ×
biological replicate (`condition`, `replicate`, `path`); at least 3 replicates
per condition are needed for replicate-level tests.

## **`CSAT.ipynb`** + `CSAT_utils.py` — saturation concentration from single-cell expression (nuclear intensity = total, nucleoplasm intensity = dilute phase).
  Three independent estimates with hierarchical-bootstrap CIs and a reliability
  verdict each: onset (logistic fit of foci presence), dilute-phase buffering
  (segmented log-log fit vs. straight line by BIC, buffering exponent β) and the
  lever rule (fraction of signal in foci = A·(1 − c_sat/c_total)). Ratio to the
  reference condition and replicate-level Welch test on log c_sat.

## **`PCA_v2.ipynb`** + `PCA_v2_utils.py` — PCA of all foci metrics (same `data` format as `METRICS.ipynb`)
- Log transform and z-scoring
- feature correlations and redundant pairs
- scree plot with parallel analysis (components above
  chance)
- loadings and correlation circle
- PCA map with replicate centroids and
  hierarchical-bootstrap confidence ellipses
- PERMANOVA / PERMDISP at cell and replicate level (exact permutations)
- distance from the reference centroid
- PC-score SuperPlots
- expression check
- within-condition outliers

## **`UMAP_v2.ipynb`** + `UMAP_v2_utils.py` — UMAP map and cell states (same `data` format and feature preparation as `PCA_v2.ipynb`).
- Settings sweep with trustworthiness and seed stability
- condition maps with density contours, each feature as a colour on the map
- KMeans clusters found in feature space (not on
  UMAP coordinates) with k chosen by silhouette and bootstrap stability (ARI)
  and a warning when cells form a continuum
- cluster profiles and most typical
  cells (image IDs), and cluster shares per condition tested on replicate
  fractions (Welch) with Fisher / chi-square as descriptive cell-level tests.

At the top of each notebook set `DATA_DIR` and the `data` list — one entry per
condition × biological replicate, with its `condition`, `replicate` and the
`path` to its `aggregated_data.csv`, e.g.
`{"condition": "WT", "replicate": "1", "path": f"{DATA_DIR}/..."}`.

## Installation

### Fiji
Install [Fiji](https://fiji.sc/). The `FIJI/` scripts need no extra plugins or
Python packages.

### Python
Use Python 3.13 (the tested version, see below); 3.11 is the minimum, because
the config files are read with the built-in `tomllib`.

```bash
git clone https://github.com/Lopatushka/LLPS.git
cd LLPS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or, with conda (installs the tested Python version and the same pinned
packages from `requirements.txt`):

```bash
conda env create -f environment.yml
conda activate foci
```

Jupyter (or VS Code with the Jupyter extension) is needed to run the notebooks.

### Tested environment

Results were produced with the following versions.

| Component | Version |
|---|---|
| Python | 3.13.13 (conda env, WSL2 / Linux) |
| Python packages | exact versions pinned in `requirements.txt` |
| Fiji | 2.16.1 (ImageJ2 2.16.0) |
| ImageJ | 1.54p |
| Jython (Fiji script engine) | 2.7.4 |
| OS for Fiji | Windows 64-bit |

## Usage

### Parameters

All segmentation parameters are in a config file next to each script
(`FOCI/config_outsu.toml`, `FOCI/config_gmm.toml`; used by default). Each parameter is commented
in the file. Every run copies the config it used into the output folder as
`run_config.toml`, so each results folder records how it was produced.

To try other settings (e.g. per microscope or for a sensitivity analysis), copy
the file, edit the copy and pass it with `--config`. Every key must be present;
misspelled or missing keys stop the run with an error.

| Parameter | `config_outsu.toml` | `config_gmm.toml` | Meaning | Notes |
|---|---|---|---|---|
| `rolling_ball_radius` | `0` | `0` | Rolling-ball background subtraction radius, px; `0` = off | Applied before smoothing. When > 0, `mean/max/min_intensity` are measured on the background-subtracted image; `partition_coefficient` always on the raw one |
| `sigma` | `0.6` | — | Gaussian smoothing before thresholding, px | |
| `smooth_sigma` | — | `1.0` | Gaussian smoothing before the GMM fit, px | |
| `threshold_factor` | `1.0` | — | Foci mask = smoothed image > Otsu threshold × `threshold_factor` | Otsu threshold is computed on the nucleus (non-zero pixels). `summary.csv` records `otsu_threshold`, `threshold_factor` and the `threshold` used |
| `n_components` | — | `4` | Number of Gaussian components fitted to the pixel intensities | Recorded in `summary.csv` |
| `signal_components` | — | `[4]` | Components used as foci signal, numbered 1…`n_components` by increasing mean intensity | `[4]` = brightest only, `[3, 4]` = two brightest. Recorded in `summary.csv` |
| `probability_threshold` | — | `0.5` | A pixel is foci if its summed probability of the signal components ≥ this value | Range (0, 1] |
| `n_init` | — | `10` | Number of GMM initialisations; the best fit is kept | |
| `random_state` | — | `42` | Seed for pixel subsampling and GMM initialisation | Same seed → same result |
| `max_fit_pixels` | — | `150000` | Fit the GMM on a random subsample of this many pixels; `0` = all pixels | Faster for large images |
| `fill_holes` | — | `true` | Fill holes inside foci masks | |
| `min_area` | `7` | `7` | Smallest object / focus kept, px | Smaller watershed fragments are merged into the touching focus; a focus split only into such fragments is kept unsplit |
| `segmentation` | `"watershed"` | `"watershed"` | How the foci mask is split into ROIs: `"watershed"` (distance transform + watershed, splits touching foci) or `"connected"` (connected regions, 8-connectivity) | Watershed splits are counted in `fraction_touching` |
| `min_distance` | `3` | `3` | Minimum distance between watershed seeds (peaks of the distance transform), px | Used only with `segmentation = "watershed"` |
| `save_figures` | `true` | — | Save `panels/<image>_six_panels.png` and `histograms/<image>_histograms.png` for every image | |
| `hist_bins` | — | `200` | Number of bins of the histogram plots | |
| `hist_percentile` | — | `99.99` | Upper intensity percentile shown in the histogram plots | |
| `figure_dpi` | `300` | `180` | Resolution of the saved figures | |

— = the parameter does not exist for that script.

### Running

```bash
cd FOCI              # from the repository root (LLPS/)
python outsu.py --input <tif_folder> --output <results>         # uses config_outsu.toml
python outsu.py -i <tif_folder> -o <results> -c my_config.toml  # other parameters
python agg.py -i <results>/roi_measurements -o <results>       # -> aggregated_data.csv
```

```bash
cd FOCI              # from the repository root (LLPS/)
python gmm.py --input <tif_folder> --output <results>           # uses config_gmm.toml
python agg.py -i <results>/roi_measurements -o <results>
```

```bash
cd FOCI              # from the repository root (LLPS/)
python mfi_nuclei.py -i <tif_folder> -o <results>              # -> MFI_nuclei.csv
```

Run the scripts from `FOCI/` (or give their path, e.g. `python FOCI/outsu.py`);
the default config is found next to the script, wherever you run it from.
`--input` and `--output` are required for every script.
Each script's `--help` explains its arguments, the expected input and every
output file, e.g. `python outsu.py --help`.

Then open `STATISTICS/METRICS.ipynb` (main comparison), `CSAT.ipynb`,
`PCA_v2.ipynb` or `UMAP_v2.ipynb`, point the `data` list at the
`aggregated_data.csv` files and run the cells.

### Measured features

`outsu.py` and `gmm.py` write the same columns, in the same order, to
`roi_measurements/<image>_ROIs.csv` (one row per focus, computed with
`skimage.measure.regionprops_table`) and to `summary/summary.csv` (one row per
cell). `agg.py` (in `FOCI/`, for both methods) writes `aggregated_data.csv`
(one row per cell).

In the file columns: ✓ = the column is in that file; *median* = median over the
foci of the cell of the per-focus value in `_ROIs.csv`; *copied* = same value
as in `summary.csv`; — = not in that file. Cells without foci are kept in
`summary.csv` and `aggregated_data.csv` (`n_rois` = `n_foci` = 0) with empty
foci features.

| Category | Metric | `_ROIs.csv` (per focus) | `summary.csv` (per cell) | `aggregated_data.csv` (per cell) | Meaning | Units | Notes |
|---|---|---|---|---|---|---|---|
| ID | `image_id` | ✓ | ✓ | ✓ | Image (= nucleus) the row belongs to: file name without extension (`img3.tif` → `img3`) | — | |
| ID | `label` | ✓ | — | — | Number of the focus within its image (1…N); its pixel value in the labelled image | — | Restarts at 1 in every image, so a focus is identified by (`image_id`, `label`) |
| Position | `centroid-0`, `centroid-1` | ✓ | — | — | Row (y) and column (x) of the focus centre | px | Not aggregated by `agg.py` |
| Number | `n_rois` / `n_foci` | — | `n_rois` | `n_foci` | Number of foci in the cell | count | 0 for cells without foci |
| Number | `foci_density` | — | ✓ | copied | `n_rois` / `nucleus_area_px`: foci per pixel of nucleus | foci / px | A foci count that does not grow with nucleus size; ×1000 for foci per 1000 px |
| Size | `area` | ✓ | — | median | Number of pixels in the focus | px | |
| Size | `equivalent_diameter_area` | ✓ | — | median | Diameter of a circle with the same area, √(4·area/π) | px | |
| Size | `perimeter` | ✓ | — | median | Length of the focus boundary | px | Approximate; unreliable for foci of a few pixels |
| Size | `total_area_px` | — | ✓ | — | Summed area of all foci of the cell | px | |
| Size | `area_fraction` | — | ✓ | copied | Foci area / nucleus area | 0–1 | |
| Size | `size_gini` | — | ✓ | copied | Gini coefficient of the foci areas of the cell, rescaled by n/(n−1): 0 = all foci the same size, 1 = one focus holds all the foci area | 0–1 | The rescaling gives the 0–1 range for any number of foci. Empty for cells with < 2 foci; noisy with few foci, so filter on `n_rois` if needed |
| Shape | `eccentricity` | ✓ | — | median | Elongation of the fitted ellipse: 0 = circle, → 1 = elongated | 0–1 | |
| Shape | `solidity` | ✓ | — | median | Area / area of the convex hull: 1 = convex, lower = irregular or indented outline | 0–1 | |
| Intensity | `mean_intensity` | ✓ | — | median | Mean pixel intensity of the focus (MFI) | grey values | Measured on the background-subtracted image when `rolling_ball_radius > 0`, otherwise on the raw image (never on the smoothed one) |
| Intensity | `max_intensity` | ✓ | — | median | Brightest pixel of the focus | grey values | As `mean_intensity` |
| Intensity | `min_intensity` | ✓ | — | median | Dimmest pixel of the focus | grey values | As `mean_intensity` |
| Partitioning | `partition_coefficient` | ✓ | — | median | Mean raw intensity of the focus / mean raw intensity of the nucleoplasm: how many times more concentrated the protein is in the focus | ratio | Always on the raw image, because the rolling ball would remove the nucleoplasm signal it is divided by. Independent of expression level |
| Partitioning | `fraction_signal_in_foci` | — | ✓ | copied | Summed raw intensity of all foci / summed raw intensity of the nucleus | 0–1 | Independent of expression level |
| Nucleus / expression | `nucleus_area_px` | — | ✓ | — | Nucleus area (non-zero pixels) | px | Calibrated area: `MFI_nuclei.csv` from `mfi_nuclei.py` |
| Nucleus / expression | `nucleus_mean_intensity` | — | ✓ | copied | Mean raw intensity of the nucleus (total concentration) | grey values | |
| Nucleus / expression | `nucleoplasm_mean_intensity` | — | ✓ | copied | Mean raw intensity of the nucleoplasm = nucleus minus all foci and a 2 px ring around them (dilute phase) | grey values | |
| Spatial | `nn_distance` | ✓ | — | median | Distance from the focus centroid to the centroid of the nearest other focus in the same nucleus | px | Empty if the nucleus has only one focus |
| Spatial | `nn_gap` | ✓ | — | median | Edge-to-edge distance to the nearest other focus: shortest distance between the pixel centres of the two foci minus 1 (touching = 0, one background pixel between = 1) | px | Independent of focus size, unlike `nn_distance`. Empty if the nucleus has only one focus |
| Spatial | `mean_nn_distance` | — | ✓ | — | Mean `nn_distance` of the cell | px | Empty for cells with < 2 foci |
| Spatial | `clark_evans_ratio` | — | ✓ | copied | `mean_nn_distance` / mean expected if the same number of foci were placed at random in that nucleus: < 1 = clustered, ≈ 1 = random, > 1 = evenly spaced | ratio | Corrects for nucleus size and foci count. Expected value with Donnelly's edge correction, 0.5·√(A/n) + (0.0514 + 0.041/√n)·P/n (A, P = nucleus area and perimeter), because foci near the nuclear edge have no neighbours outside it. Uses centre-to-centre distances on purpose: the random expectation is only defined for points. Foci are not points and cannot sit closer than about their radii, so large or densely packed foci push the ratio above 1 even when placed at random. Empty for cells with < 2 foci |
| Spatial | `fraction_touching` | — | ✓ | copied | Fraction of the cell's foci whose `nn_gap` ≤ 1 px (`TOUCHING_GAP_PX` in the scripts): foci in contact or separated by one pixel | 0–1 | With `segmentation = "watershed"` this includes the parts of every split focus, so it also shows how much watershed splitting happened. 0 for a cell with one focus, empty without foci |

## Citation

If you use this code in your research, please cite this repository:

```bibtex
@software{foci_llps,
  author = {Elena Lopatukhina},
  title  = {Foci: nuclear foci segmentation and analysis},
  year   = {2026},
  url    = {https://github.com/Lopatushka/LLPS}
}
```

A publication describing this work is in preparation; this section will be
updated with the reference once it is available.

## License

This project is licensed — see [LICENSE](LICENSE) for details.
