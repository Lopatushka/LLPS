# Foci — nuclear foci segmentation and analysis

Tools for segmenting nuclei and nuclear foci in fluorescence microscopy images,
measuring foci morphology and intensity, and comparing experimental conditions
with multivariate statistics (PCA, UMAP, PERMANOVA).

> Status: work in progress

## Workflow

```
raw multichannel .tif
   │  FIJI/nuclei_manual_segment.py   (Fiji)  → one .tif per nucleus + ROI .zip
   │  FIJI/substruct_background.py    (Fiji)  → *_bg_subtracted.tif   (optional)
   │  FIJI/MFI_nuclei.py              (Fiji)  → MFI_nuclei.csv        (optional)
   ▼
single-nucleus .tif images
   │  FOCI_OUTSU/outsu.py  or   FOCI_GMM/gmm.py   → per-image foci tables
   │  FOCI_OUTSU/agg.py   or   FOCI_GMM/agg.py   → aggregated_data.csv (one row per cell)
   ▼
aggregated_data.csv for each condition
   │  STATISTICS/PCA.ipynb, STATISTICS/UMAP.ipynb
   ▼
plots and statistical tests
```

## Folder structure

```
Foci/
├── FIJI/              Jython scripts, run inside Fiji/ImageJ
├── FOCI_OUTSU/        Otsu + watershed foci segmentation
├── FOCI_GMM/          Gaussian-mixture-model foci segmentation
├── STATISTICS/        PCA / UMAP analysis notebooks and helpers
└── requirements.txt   Python dependencies
```

### `FIJI/` — Fiji/ImageJ scripts

Run these from Fiji's Script Editor (language: Python/Jython). They work on the
images currently open in Fiji and ask for an output folder.

| Script | What it does |
|---|---|
| `nuclei_manual_segment.py` | Asks for microscope type (Airyscan / NSpark), DAPI channel and measurement channel. For each open image you draw nuclear ROIs on the DAPI channel. Each nucleus is saved as a separate `.tif` (`C<channel>_<image>_ROI_<nn>_....tif`) and the ROIs as `<image>_rois.zip`. |
| `substruct_background.py` | Rolling-ball background subtraction (radius set in a dialog) on all open images; saves `<image>_bg_subtracted.tif`. |
| `MFI_nuclei.py` | Measures area and mean intensity of every open image and saves `MFI_nuclei.csv`. |
| `opener.py` | Opens all files in a folder whose name contains a given pattern and has a given extension. |

### `FOCI_OUTSU/` — Otsu threshold segmentation

- **`outsu.py`** — for every `.tif`/`.tiff` in the input folder: optional
  rolling-ball background subtraction (`rolling_ball_radius`, `0` = off, the
  default) → Gaussian smoothing (`sigma`) → Otsu threshold × `threshold_factor` → small-object removal
  (`min_area`) → distance transform + watershed to split
  touching foci. Parameters are set in `FOCI_OUTSU/config.toml`. Outputs go
  into the same subfolders as `gmm.py`: `roi_measurements/` (`<image>_ROIs.csv`,
  one row per focus, see *Measured foci features* below), `summary/`
  (`all_ROIs.csv`, and `summary.csv` with the Otsu threshold, the threshold used,
  number of foci and foci area per image), `binary_masks/`, `labeled_rois/`
  (after watershed), `backgrounds/`, `corrected_images/`, `smoothed_images/`,
  and — if `save_figures = true` — `panels/` (`<image>_six_panels.png`: original,
  background, corrected, mask, distance transform, numbered watershed ROIs) and
  `histograms/` (`<image>_histograms.png`: intensity histogram with the threshold).
  With `rolling_ball_radius > 0` the intensities are measured on the
  background-subtracted image, otherwise on the raw image.
- **`agg.py`** — reads all `*_ROIs.csv` files in `roi_measurements/` and writes
  `aggregated_data.csv` with one row per cell: `n_foci` plus the median of each
  foci feature. Cells with zero foci are kept (`n_foci = 0`).

### `FOCI_GMM/` — Gaussian mixture model segmentation

- **`gmm.py`** — fits a GMM (`n_components`, 4 by default) to pixel intensities
  of each image; pixels whose probability of belonging to the brightest
  component(s) (`signal_components`) is at least `probability_threshold` form
  the foci mask.
  Optional rolling-ball background subtraction (`rolling_ball_radius`, `0` =
  off) and Gaussian smoothing. With `rolling_ball_radius > 0` the foci
  intensities (`mean/max/min_intensity`) are measured on the
  background-subtracted image, otherwise on the raw image. Outputs
  into subfolders: `histograms/`, `panels/`, `backgrounds/`,
  `corrected_images/`, `smoothed_images/`, `binary_masks/`, `labeled_rois/`,
  `roi_measurements/` (`*_ROIs.csv`) and `summary/`.
  Parameters are set in `FOCI_GMM/config.toml`.
- **`agg.py`** — aggregates the `*_ROIs.csv` tables into `aggregated_data.csv`
  (one row per image).
- **`diff43.py`** — uses the `smoothed_images/` from a previous `gmm.py` run to
  compare masks built from component 4 only vs. components 3 + 4.

### `STATISTICS/` — condition comparison

- **`PCA.ipynb`** + `PCA_utils.py` — PCA on standardised per-cell features,
  Spearman correlation heatmap, violin plots, bootstrapped centroid confidence
  regions, PERMANOVA / PERMDISP between samples, centroid distance tree, outlier
  detection.
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

All segmentation parameters are in a `config.toml` file next to each script
(`FOCI_OUTSU/config.toml`, `FOCI_GMM/config.toml`). Each parameter is commented
in the file. Every run copies the config it used into the output folder as
`run_config.toml`, so each results folder records how it was produced.

To try other settings (e.g. per microscope or for a sensitivity analysis), copy
the file, edit the copy and pass it with `--config`. Every key must be present;
misspelled or missing keys stop the run with an error.

### Running

```bash
cd Foci/FOCI_OUTSU
python outsu.py --input <tif_folder> --output <results>         # uses config.toml
python outsu.py -i <tif_folder> -o <results> -c my_config.toml  # other parameters
python agg.py -i <results>/roi_measurements -o <results>       # -> aggregated_data.csv
```

```bash
cd Foci/FOCI_GMM
python gmm.py --input <tif_folder> --output <results>
python agg.py -i <results>/roi_measurements -o <results>
python diff43.py -i <results> -o <comparison_results>   # optional: G4 vs G3+G4
```

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

Intensities are measured on the background-subtracted image when
`rolling_ball_radius > 0`, otherwise on the raw image (not on the smoothed one).

`agg.py` (both methods) writes `aggregated_data.csv` with one row per cell:
`image_id`, `n_foci` (number of foci in the cell) and the median of each feature
above over all foci of that cell (the centroids are not aggregated). Cells
without foci have `n_foci = 0` and empty feature values.
