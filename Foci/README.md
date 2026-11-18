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
   │  FOCI_OUTSU/foci.py   or   FOCI_GMM/gmm.py   → per-image foci tables
   │  FOCI_OUTSU/agg.py    or   FOCI_GMM/agg.py   → aggregated_data.csv (one row per cell)
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

- **`foci.py`** — for every `.tif`/`.tiff` in the input folder: Gaussian smoothing
  (`sigma=0.6`) → Otsu threshold × `threshold_factor` (3 by default in `main`) →
  small-object removal (`min_area=2`) → distance transform + watershed to split
  touching foci. For each image it writes:
  - `<image>_segmentation.csv`: one row per focus (`area`, `centroid`,
    `equivalent_diameter_area`, `eccentricity`, `solidity`, `perimeter`,
    `mean/max/min_intensity`, `circularity`, `image_id`)
  - `<image>_segmentation.png`: mask, distance transform and watershed overlay
- **`agg.py`** — reads all `*_segmentation.csv` files and writes
  `aggregated_data.csv` with one row per cell: `n_foci` plus the median of each
  foci feature. Cells with zero foci are kept (`n_foci = 0`).

### `FOCI_GMM/` — Gaussian mixture model segmentation

- **`gmm.py`** — fits a GMM (`N_COMPONENTS = 4`) to pixel intensities of each
  image; pixels whose probability of belonging to the brightest component(s)
  (`SIGNAL_COMPONENTS`) exceeds `PROBABILITY_THRESHOLD` form the foci mask.
  Optional rolling-ball background subtraction and Gaussian smoothing. Outputs
  into subfolders: `histograms/`, `panels/`, `backgrounds/`,
  `corrected_images/`, `smoothed_images/`, `binary_masks/`, `labeled_rois/`,
  `roi_measurements/` (`*_ROIs.csv`) and `summary/`.
  Parameters are set in the `CONFIG` block at the top of the file.
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
Python 3.10+ is recommended.

```bash
git clone https://github.com/Lopatushka/LLPS.git
cd LLPS/Foci
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Jupyter (or VS Code with the Jupyter extension) is needed to run the notebooks.

## Usage

The Python scripts ask for the input and output folders interactively:

```bash
cd Foci/FOCI_OUTSU
python foci.py      # input: folder with single-nucleus .tif images
python agg.py       # input: the foci.py output folder
```

```bash
cd Foci/FOCI_GMM
python gmm.py
python agg.py       # input: the roi_measurements/ folder
```

Then open `STATISTICS/PCA.ipynb` or `STATISTICS/UMAP.ipynb`, point the `data`
list at the `aggregated_data.csv` files and run the cells.
