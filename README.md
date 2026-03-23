# LLPS — nuclei & foci segmentation helpers (Fiji/ImageJ + Python)

This repo (master branch) includes:
- `nuclei_segmentation.py` — nuclei segmentation / ROI generation (Python)
- `foci_segmentation.py` — Fiji/ImageJ workflow for foci detection via ThunderSTORM + export
- `foci_stats.py` — merges nuclei + foci CSV tables and computes Spearman correlations
- `foci_graphs.ipynb` — plotting / graphs notebook
- `FRAP.py` — ROIs generation and exporting for FRAP analysis
- `FRAP_graphs.ipynb` — plotting / graphs notebook
- `examples/` — directory with example input/output files

> Status: work-in-progress

# Foci analysis
## Typical workflow

1) **Segment nuclei** on microscopy images using *nuclei_segmentation.py* macros for **Fiji** 
→ produce nucleus ROIs, per-nucleus measurements (e.g., `Area`, `Mean`).

2) **Detect foci** (optionally restricted to nuclei ROIs) using *foci_segmentation.py* macros for **Fiji**  
→ export foci list per image to CSV (e.g., ThunderSTORM table export).

3) **Aggregate and run statistics** using *foci_stats.py* in **Python** 
→ merge nuclei + foci summaries by file key and export `results.csv` + `spearman_pairs.csv`.

3) **Make statistical data analysis and plot graphs** using *FRAP_graphs.ipynb* notebook in **Python** 

# FRAP analysis
## Typical workflow



## Requirements

### ImageJ / Fiji
- Fiji (ImageJ distribution)
- ThunderSTORM plugin (if you use ThunderSTORM for foci)
- ImageJ / IJ libraries (for running scripts/macros) :contentReference[oaicite:3]{index=3}

### Python
- Python 3.x
- `pandas`
- `numpy`
- `scipy`
- `ij`
- `matplotlib`
- `pathlib`


Installation:
```bash
pip install pandas scipy
```


## Installation

Clone the repo:

```bash
git clone https://github.com/Lopatushka/LLPS.git
cd LLPS
```