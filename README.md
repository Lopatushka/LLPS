# LLPS — nuclei & foci segmentation helpers (Fiji/ImageJ + Python)

This repository contains small scripts used for image analysis around **LLPS / foci-like structures**, focusing on:

- **Nuclei segmentation** (Python)
- **Foci segmentation & quantification** (Fiji/ImageJ workflow script)

> Status: work-in-progress / lab utility code (no packaged CLI yet). :contentReference[oaicite:1]{index=1}

---

## Contents

- `nuclei_segmentation.py`  
  Script for generating nuclei masks / ROIs from microscopy images.

- `foci_segmentation.ijm.ijm.py`  
  Fiji/ImageJ workflow script for foci detection/segmentation and exporting results (e.g., tables/CSV).  
  *(Filename suggests it may be an `.ijm` macro exported/embedded via Python/Jython, or a macro-like script — adjust as needed.)*
---

## Typical workflow

1. **Segment nuclei** on your images  
   → produce nuclei masks and ROIs.

2. **Detect/segment foci inside nuclei** in Fiji/ImageJ  
   → run the foci script, restrict analysis to nuclei ROIs, export results.

---

## Requirements
- Fiji (ImageJ distribution)
- Plugin ThunderSTORM
- IJ library for Phyton
---

## Installation

Clone the repo:

```bash
git clone https://github.com/Lopatushka/LLPS.git
cd LLPS