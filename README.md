# Foci — nuclear foci segmentation and analysis

Tools for segmenting nuclei and nuclear foci in fluorescence microscopy images,
measuring foci morphology and intensity, and comparing experimental conditions
with multivariate statistics (PCA, UMAP, PERMANOVA).

> Status: work in progress

## Workflow

```
raw multichannel .tif
   │  FIJI/nuclei_manual_segment.py   (Fiji)  → one .tif per nucleus + ROI .zip
   ▼
single-nucleus .tif images
   │  FOCI/mfi_nuclei.py                         → MFI_nuclei.csv (optional)
   │  FOCI/outsu.py  or  FOCI/gmm.py             → per-image foci tables
   │  FOCI/agg.py   (both methods)               → aggregated_data.csv (one row per cell)
   ▼
aggregated_data.csv for each condition
   │  STATISTICS/PCA.ipynb, STATISTICS/UMAP.ipynb
   ▼
plots and statistical tests
```

## Folder structure

```
Foci/
├── FIJI/                  Jython scripts, run inside Fiji/ImageJ
├── FOCI/                  Python scripts for foci segmentation and measurement
│   ├── outsu.py           Otsu threshold foci segmentation   (config_outsu.toml)
│   ├── gmm.py             Gaussian-mixture-model foci segmentation (config_gmm.toml)
│   ├── config_outsu.toml  parameters of outsu.py
│   ├── config_gmm.toml    parameters of gmm.py
│   ├── agg.py             aggregates the foci tables of either method, one row per cell
│   └── mfi_nuclei.py      nuclear area and mean intensity (MFI) of every image
├── STATISTICS/            PCA / UMAP / METRICS / CSAT analysis notebooks and helpers
├── OLD/                   earlier code, no longer used by the workflow
└── requirements.txt       Python dependencies
```

### `FIJI/` — Fiji/ImageJ scripts

Run these from Fiji's Script Editor (language: Python/Jython). They work on the
images currently open in Fiji and ask for an output folder.

| Script | What it does |
|---|---|
| `nuclei_manual_segment.py` | Asks for microscope type (Airyscan / NSpark), DAPI channel and measurement channel. For each open image you draw nuclear ROIs on the DAPI channel. Each nucleus is saved as a separate `.tif` (`C<channel>_<image>_ROI_<nn>_....tif`) and the ROIs as `<image>_rois.zip`. |
| `opener.py` | Opens, in alphabetical order, all files in a folder whose name contains a given pattern (case-insensitive; empty = any) and has a given extension. Optionally also searches subfolders. Unreadable files are skipped and listed in the Log. |
| `closer.py` | Closes all open images after one confirmation dialog, discarding unsaved changes (no "Save?" prompt per image). Tables and the Log stay open. |

### `FOCI/outsu.py` — Otsu threshold segmentation

Both `outsu.py` and `gmm.py` expect single-nucleus images with everything outside
the nucleus set to 0: the non-zero pixels are the nucleus. Foci are only searched
inside it, and `area_fraction` in `summary.csv` is foci area / nucleus area.

- **`outsu.py`** — for every `.tif`/`.tiff` in the input folder: optional
  rolling-ball background subtraction (`rolling_ball_radius`, `0` = off, the
  default) → Gaussian smoothing (`sigma`) → Otsu threshold (computed on the nucleus, i.e. the non-zero pixels) × `threshold_factor` → small-object removal
  (`min_area`) → foci ROIs: `segmentation = "watershed"` (default; distance
  transform + watershed to split touching foci, seeds `min_distance` apart;
  fragments smaller than `min_area` are merged into the touching focus; a focus
  split only into such fragments is kept unsplit) or
  `"connected"` (connected regions of the mask); ROIs smaller than `min_area` are then removed. Parameters are set in `config_outsu.toml`. Outputs go
  into the same subfolders as `gmm.py`: `roi_measurements/` (`<image>_ROIs.csv`,
  one row per focus, see *Measured foci features* below), `summary/`
  (`all_ROIs.csv`, and `summary.csv` with the Otsu threshold, the threshold used,
  number of foci, nucleus area and foci area per image), `binary_masks/`, `labeled_rois/`,
  `backgrounds/`, `corrected_images/`, `smoothed_images/`,
  and — if `save_figures = true` — `panels/` (`<image>_six_panels.png`: original,
  background, corrected, mask, distance transform, segmented ROIs) and
  `histograms/` (`<image>_histograms.png`: intensity histogram with the threshold).
  With `rolling_ball_radius > 0` the intensities are measured on the
  background-subtracted image, otherwise on the raw image.

### `FOCI/gmm.py` — Gaussian mixture model segmentation

- **`gmm.py`** — fits a GMM (`n_components`, 4 by default) to pixel intensities
  of each image; pixels whose probability of belonging to the brightest
  component(s) (`signal_components`) is at least `probability_threshold` form
  the foci mask. The mask is split into ROIs by `segmentation`: `"watershed"`
  (default; distance transform + watershed, seeds `min_distance` apart;
  fragments smaller than `min_area` are merged into the touching focus; a focus
  split only into such fragments is kept unsplit) or
  `"connected"` (connected regions); ROIs smaller than `min_area` are then removed.
  Optional rolling-ball background subtraction (`rolling_ball_radius`, `0` =
  off) and Gaussian smoothing. With `rolling_ball_radius > 0` the foci
  intensities (`mean/max/min_intensity`) are measured on the
  background-subtracted image, otherwise on the raw image. Outputs
  into subfolders: `histograms/`, `panels/`, `backgrounds/`,
  `corrected_images/`, `smoothed_images/`, `binary_masks/`, `labeled_rois/`,
  `roi_measurements/` (`*_ROIs.csv`) and `summary/`.
  Parameters are set in `config_gmm.toml`.

### `FOCI/agg.py` — aggregation (both methods)

Reads all `*_ROIs.csv` files in the `roi_measurements/` folder of an `outsu.py`
or `gmm.py` run and writes `aggregated_data.csv` with one row per cell: `n_foci`
plus the median of each foci feature, and the per-cell
`foci_density`, `area_fraction`, `nucleus_mean_intensity`, `nucleoplasm_mean_intensity`, `fraction_signal_in_foci`, `size_gini`, `clark_evans_ratio` and `fraction_touching` from `summary/summary.csv` of the same run. Cells with
zero foci are kept (`n_foci = 0`).

### `FOCI/mfi_nuclei.py` — nuclear area and mean intensity

For every
`.tif`/`.tiff` in the input folder it measures the nucleus (pixels > 0) and
writes `MFI_nuclei.csv` with `image_id`, `Filename`, `Area` (calibrated from the
TIFF resolution tags, e.g. micron^2), `unit`, `Area_px` and `Mean`. `Area_px` and
`Mean` equal `nucleus_area_px` and `nucleus_mean_intensity` in `summary.csv` of
`outsu.py` / `gmm.py` for the same images. An existing `MFI_nuclei.csv` is never
overwritten (`MFI_nuclei_1.csv`, `_2`, ...).

### `STATISTICS/` — condition comparison

- **`METRICS.ipynb`** + `METRICS_utils.py` — **main result**: per-metric
  comparison of conditions. SuperPlots (cells coloured by replicate, replicate
  medians), Welch t-test on replicate medians (cell-level Mann–Whitney as
  descriptive), median ratio with hierarchical-bootstrap 95 % CI, Cliff's delta,
  Benjamini–Hochberg FDR over all metrics × comparisons, effect-size heatmap, and
  an expression-level check (Spearman with `nucleus_mean_intensity`,
  expression-matched re-analysis). The `data` list has one entry per condition ×
  biological replicate (`condition`, `replicate`, `path`); at least 3 replicates
  per condition are needed for replicate-level tests.
- **`CSAT.ipynb`** + `CSAT_utils.py` — saturation concentration from single-cell
  expression (nuclear intensity = total, nucleoplasm intensity = dilute phase).
  Three independent estimates with hierarchical-bootstrap CIs and a reliability
  verdict each: onset (logistic fit of foci presence), dilute-phase buffering
  (segmented log-log fit vs. straight line by BIC, buffering exponent β) and the
  lever rule (fraction of signal in foci = A·(1 − c_sat/c_total)). Ratio to the
  reference condition and replicate-level Welch test on log c_sat.
- **`PCA_v2.ipynb`** + `PCA_v2_utils.py` — PCA of all foci metrics (same `data`
  format as `METRICS.ipynb`). Log transform and z-scoring, feature correlations
  and redundant pairs, scree plot with parallel analysis (components above
  chance), loadings and correlation circle, PCA map with replicate centroids and
  hierarchical-bootstrap confidence ellipses, PERMANOVA / PERMDISP at cell and
  replicate level (exact permutations), distance from the reference centroid,
  PC-score SuperPlots, expression check, within-condition outliers.
- **`PCA.ipynb`** + `PCA_utils.py` — PCA on standardised per-cell features,
  Spearman correlation heatmap, violin plots, bootstrapped centroid confidence
  regions, PERMANOVA / PERMDISP between samples, centroid distance tree, outlier
  detection.
- **`UMAP_v2.ipynb`** + `UMAP_v2_utils.py` — UMAP map and cell states (same
  `data` format and feature preparation as `PCA_v2.ipynb`). Settings sweep with
  trustworthiness and seed stability, condition maps with density contours, each
  feature as a colour on the map, KMeans clusters found in feature space (not on
  UMAP coordinates) with k chosen by silhouette and bootstrap stability (ARI)
  and a warning when cells form a continuum, cluster profiles and most typical
  cells (image IDs), and cluster shares per condition tested on replicate
  fractions (Welch) with Fisher / chi-square as descriptive cell-level tests.
- **`UMAP.ipynb`** + `UMAP_utils.py` — UMAP embedding (with an `n_neighbors`
  sweep), KMeans / agglomerative clustering, chi-square test of cluster
  composition across samples, PERMANOVA within clusters, centroid CIs.

At the top of each notebook set `root_dir` and the `data` list — one entry per
condition with a sample label and the path to its `aggregated_data.csv`. The
notebook concatenates them into one table with a `sample` column
(`CONCATENATE_DATA`, `CLEAN_DATA`, `SAVE_CONCATENATED_DATA` flags).

## Installation

### Fiji
Install [Fiji](https://fiji.sc/). The `FIJI/` scripts need no extra plugins or
Python packages.

### Python
Use Python 3.13 (the tested version, see below); 3.11 is the minimum, because
the config files are read with the built-in `tomllib`.

```bash
git clone https://github.com/Lopatushka/LLPS.git
cd LLPS/Foci
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

### Running

```bash
cd Foci/FOCI          # from the repository root (LLPS/)
python outsu.py --input <tif_folder> --output <results>         # uses config_outsu.toml
python outsu.py -i <tif_folder> -o <results> -c my_config.toml  # other parameters
python agg.py -i <results>/roi_measurements -o <results>       # -> aggregated_data.csv
```

```bash
cd Foci/FOCI          # from the repository root (LLPS/)
python gmm.py --input <tif_folder> --output <results>           # uses config_gmm.toml
python agg.py -i <results>/roi_measurements -o <results>
```

```bash
cd Foci/FOCI          # from the repository root (LLPS/)
python mfi_nuclei.py -i <tif_folder> -o <results>              # -> MFI_nuclei.csv
```

Run the scripts from `Foci/FOCI/` (or give their path, e.g. `python Foci/FOCI/outsu.py`);
the default config is found next to the script, wherever you run it from.
`--input` and `--output` are required for every script.
Each script's `--help` explains its arguments, the expected input and every
output file, e.g. `python outsu.py --help`.

Then open `STATISTICS/PCA.ipynb` or `STATISTICS/UMAP.ipynb`, point the `data`
list at the `aggregated_data.csv` files and run the cells.

### Measured foci features

`outsu.py` and `gmm.py` (`roi_measurements/<image>_ROIs.csv`) write the same columns, in the same order,
one row per focus. They are computed with `skimage.measure.regionprops_table`.

| Column | Meaning | Units |
|---|---|---|
| `image_id` | Image (= nucleus) the focus belongs to: the file name without its extension (`img3.tif` → `img3`) | — |
| `label` | Number of the focus within its image (1…N); its pixel value in the labelled image. Restarts at 1 in every image, so a focus is identified by (`image_id`, `label`) | — |
| `area` | Number of pixels in the focus | px |
| `centroid-0`, `centroid-1` | Row (y) and column (x) of the focus centre | px |
| `equivalent_diameter_area` | Diameter of a circle with the same area, √(4·area/π) | px |
| `eccentricity` | Elongation of the fitted ellipse: 0 = circle, → 1 = elongated | 0–1 |
| `solidity` | Area / area of the convex hull: 1 = convex, lower = irregular or indented outline | 0–1 |
| `perimeter` | Length of the focus boundary (approximate; unreliable for foci of a few pixels) | px |
| `mean_intensity` | Mean pixel intensity of the focus (MFI) | grey values |
| `max_intensity` | Brightest pixel of the focus | grey values |
| `min_intensity` | Dimmest pixel of the focus | grey values |
| `partition_coefficient` | Mean raw intensity of the focus / mean raw intensity of the nucleoplasm (nucleus minus all foci and a 2 px ring around them). How many times more concentrated the protein is in the focus; independent of expression level | ratio |
| `nn_distance` | Distance from the focus centroid to the centroid of the nearest other focus in the same nucleus. Empty if the nucleus has only one focus | px |
| `nn_gap` | Edge-to-edge distance to the nearest other focus: shortest distance between the pixel centres of the two foci minus 1, so touching foci (e.g. two parts of a watershed split) have 0 and one background pixel between them gives 1. Independent of focus size, unlike `nn_distance`. Empty if the nucleus has only one focus | px |

Intensities are measured on the background-subtracted image when
`rolling_ball_radius > 0`, otherwise on the raw image (not on the smoothed one).
`partition_coefficient` is always measured on the raw image, because the
rolling ball would remove the nucleoplasm signal it is divided by.

`summary/summary.csv` (one row per cell) also has `nucleus_area_px`,
`foci_density` (`n_rois` / `nucleus_area_px`: foci per pixel of nucleus, so a
foci count that does not grow with nucleus size; ×1000 for foci per 1000 px),
`area_fraction` (foci area / nucleus area), `nucleus_mean_intensity`,
`nucleoplasm_mean_intensity` and `fraction_signal_in_foci` (summed raw intensity
of all foci / summed raw intensity of the nucleus; 0–1, independent of
expression level) and `size_gini`: the Gini coefficient of the foci areas of
the cell, rescaled by n/(n−1) so that it ranges from 0 (all foci the same size)
to 1 (one focus holds all the foci area) for any number of foci. Empty for cells
with fewer than 2 foci; noisy with few foci, so filter on `n_rois` if needed.
`mean_nn_distance` is the mean `nn_distance` of the cell (px), and
`clark_evans_ratio` divides it by the mean expected if the same number of foci
were placed at random in that nucleus: < 1 = clustered, ≈ 1 = random,
> 1 = evenly spaced. This corrects for nucleus size and foci count. The expected
value uses Donnelly's edge correction, 0.5·√(A/n) + (0.0514 + 0.041/√n)·P/n
(A, P = nucleus area and perimeter), because foci near the nuclear edge have no
neighbours outside it. Foci are not points: they cannot sit closer than about
their radii, so large or densely packed foci push the ratio above 1 even when
placed at random. Both are empty for cells with fewer than 2 foci.
`clark_evans_ratio` uses centre-to-centre distances on purpose: its random
expectation is only defined for points. `fraction_touching` is the fraction of
the cell's foci whose `nn_gap` is ≤ 1 px (`TOUCHING_GAP_PX` in the scripts):
foci in contact or separated by one pixel. With `segmentation = "watershed"`
this includes the parts of every split focus, so it also shows how much
watershed splitting happened. 0 for a cell with one focus, empty without foci.

`agg.py` (in `Foci/FOCI/`, for both methods) writes `aggregated_data.csv` with one row per cell:
`image_id`, `n_foci` (number of foci in the cell) and the median of each feature
above over all foci of that cell (the centroids are not aggregated), plus
`foci_density`, `area_fraction`, `nucleus_mean_intensity`, `nucleoplasm_mean_intensity`, `fraction_signal_in_foci`, `size_gini`, `clark_evans_ratio` and `fraction_touching` from `summary/summary.csv`. Cells without foci have
`n_foci = 0` and empty feature values.

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
