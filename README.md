# LLPS — nuclei & foci segmentation helpers (Fiji/ImageJ + Python)

This repo (master branch) includes:
- `nuclei_segmentation.py` — nuclei segmentation / ROI generation (Python)
- `foci_segmentation.ijm.ijm.py` — Fiji/ImageJ workflow for foci detection + export
- `statisctics.py` — merges nuclei + foci CSV tables and computes Spearman correlations
- `opener.py` — helper script (opening/IO utility)
- `graphs.ipynb` — plotting / graphs notebook
- `data_examples/` — example input/output files

> Status: work-in-progress

## Typical workflow

1) **Segment nuclei** on microscopy images  
→ produce nucleus ROIs and/or per-nucleus measurements (e.g., `Area`, `Mean`).

2) **Detect foci** (optionally restricted to nuclei ROIs) in Fiji/ImageJ  
→ export foci list per image to CSV (e.g., ThunderSTORM table export).

3) **Aggregate and run statistics** in Python  
→ merge nuclei + foci summaries by file key and export `results.csv` + `spearman_pairs.csv`.

## Requirements

### ImageJ / Fiji
- Fiji (ImageJ distribution)
- ThunderSTORM plugin (if you use ThunderSTORM for foci)
- ImageJ / IJ libraries (for running scripts/macros) :contentReference[oaicite:3]{index=3}

### Python
- Python 3.x
- `pandas`
- `scipy`

Install:
```bash
pip install pandas scipy


## Installation

Clone the repo:

```bash
git clone https://github.com/Lopatushka/LLPS.git
cd LLPS