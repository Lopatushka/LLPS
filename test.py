from ij import IJ
from ij.plugin.frame import RoiManager
from ij import ImagePlus
from ij.gui import GenericDialog
from ij import WindowManager
import os
import re
import csv

def _safe_name(s):
    """Make a string safe for filenames."""
    s = str(s)
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = s.replace(" ", "_")
    return s

def close_window(title):
    win = WindowManager.getWindow(title)
    if win: win.dispose()

def foci_image(imp, rois):
    """
    Process a single image for multiple ROIs.

    imp  : ImagePlus
    rois : list of Roi objects
    p    : dict-like parameters (optional, used later)
    """
    img_name = imp.getTitle()
    img_base = _safe_name(os.path.splitext(img_name)[0])

    for i, roi in enumerate(rois):
        dup = None
        roi_name = None
        try:
            roi_name = roi.getName()
            if roi_name is None:
                roi_name = "roi_{:02d}".format(i + 1)
            roi_base = _safe_name(roi_name)

            IJ.log("Processing image: {} and ROI: {}".format(img_name, roi_name))

            # Make sure old results window doesn't interfere
            close_window("ThunderSTORM: results")

            # Set ROI and crop
            imp.setRoi(roi)
            dup = imp.duplicate()
            dup.show()
            dup.setRoi(roi)
            dup.setTitle("ROI_{:02d}_{}".format(i + 1, img_name))
            IJ.run(dup, "Clear Outside", "")
            dup.killRoi()

        except Exception as e:
            IJ.log("Error processing ROI {}: {}".format(roi_name, e))
            if dup:
                dup.close()

# --- Main ---
imp = IJ.getImage()
#print(imp)
roi_manager = RoiManager.getInstance()
rois = roi_manager.getRoisAsArray()

foci_image(imp, rois)

