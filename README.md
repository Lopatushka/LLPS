# LLPS — nuclei & foci segmentation helpers (Fiji/ImageJ + Python)

Pipeline for microscopy image processing and foci analysis.

The project processes microscopy `.tif` images together with ThunderSTORM localization `.csv` files, calculates foci parameters, filters outliers, and generates histograms.

This repo (master branch) includes 2 folders:
- `Foci` — the Python package to analyse the foci morphology using the modified ThunderSTORM algorithm.
- `FRAP` — the Python package for FRAP data analysis using the EasyFRAP algorithm.

# Project Structure

Foci/

├── auto_segmentation.py  
│   # Jython script for automatic nuclei segmentation in microscopy images.  
│   # Run inside ImageJ/Fiji.  
│   # Generates ROI masks and segmented nuclei images automatically.

├── manual_segmentation.py  
│   # Jython script for manual nuclei segmentation and correction.  
│   # Run inside ImageJ/Fiji.  
│   # Allows user-guided ROI selection.

├── foci_segmentation.py  
│   # Jython script for foci detection and segmentation.
│   # Run inside ImageJ/Fiji.  
│   # Processes microscopy images and prepares data for ThunderSTORM analysis.


├── foci_processing.py  
│   # Processes ThunderSTORM localization tables.  
│   # Calculates foci properties, filters outliers,  
│   # generates histograms, and exports processed data.

├── foci_data_aggregation.py  
│   # Aggregates foci-level measurements from multiple images/files  
│   # into summary tables for downstream analysis.

├── nuclei_data_aggregation.py  
│   # Aggregates nuclei-level measurements such as area and intensity  
│   # of the whole nucleus.

├── statistics.ipynb  
│   # Jupyter notebook for statistical analysis and visualization.  
│   # Includes exploratory plots, summary statistics,  
│   # and comparison between experimental conditions.

FRAP/

├── frap_roi.py
│
│

├── frap_analysis.ipynb
│
│

├── frap_stats.ipynb
│
│

├── frap_utils.py
│
│

> Status: work-in-progress

## Requirements

### ImageJ / Fiji
- Fiji (ImageJ distribution)
- ThunderSTORM plugin (if you use ThunderSTORM for foci detection)

### Python
- Python 3.x
- `pandas`
- `numpy`
- `scipy`
- `ij`
- `matplotlib`
- `pathlib`
- `csv`


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