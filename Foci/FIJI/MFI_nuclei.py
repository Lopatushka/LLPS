"""
MFI_nuclei.py  (Fiji / Jython script)

Measures the area and mean fluorescence intensity (MFI) of the nucleus in every
image that is currently open in Fiji and saves one row per image to
<output_dir>/MFI_nuclei.csv with the columns: Filename, Area, Area_px, Mean.

Intended input: the single-nucleus .tif images saved by nuclei_manual_segment.py
(pixels outside the nucleus are 0).

How to run: open the images in Fiji -> run this script -> choose an output folder.

NOTES / CAVEATS
  * Nucleus = pixels with value > 0 (same definition as outsu.py / gmm.py).
    Only these pixels are measured, so "Mean" equals nucleus_mean_intensity and
    "Area_px" equals the nucleus area of the Python scripts (on the same image).
    An ROI drawn on the image is ignored.
  * "Area" is in calibrated units (e.g. um^2) if the image has a spatial
    calibration, otherwise in pixels (then Area == Area_px).
  * Whatever is open is measured: raw or *_bg_subtracted.tif images give different
    results. For multi-channel / stack images only the current slice/channel is used.
  * Rows are in the order the images were opened.
  * If MFI_nuclei.csv already exists in the output folder, the table is saved as
    MFI_nuclei_1.csv, MFI_nuclei_2.csv, ... (existing files are never overwritten).
  * All measured images are closed at the end WITHOUT saving changes.
"""

from ij import IJ, WindowManager
from ij.measure import ResultsTable
import os
import traceback


def close_images(imps):
    """Close the given images. Unsaved changes are discarded (no 'Save?' prompt)."""
    for im in imps:
        if im is None:
            continue
        im.changes = False  # mark as unchanged so Fiji does not ask to save
        im.close()


def close_all_csv_tables():
    """Close all non-image windows whose title ends in '.csv'.

    After IJ.saveAs("Results", ".../MFI_nuclei.csv") the Results window is
    renamed to the file name, so this closes the table that was just saved
    (and also any other open .csv table).
    """
    windows = WindowManager.getAllNonImageWindows()
    if windows is not None:
        for w in windows:
            title = w.getTitle()
            if title.endswith(".csv"):
                w.dispose()


def nucleus_stats(imp):
    """Area and mean intensity of the nucleus (pixels > 0) of the current slice.

    Returns (area_calibrated, area_px, mean). Raises ValueError if the image
    has no pixels > 0.
    """
    # Float copy of the current slice/channel: same values for 8/16/32-bit images
    pixels = imp.getProcessor().convertToFloatProcessor().getPixels()

    count = 0
    total = 0.0
    for v in pixels:
        if v > 0:  # nucleus pixel; the 0 pixels outside the nucleus are skipped
            count += 1
            total += v

    if count == 0:
        raise ValueError("image has no non-zero (nucleus) pixels")

    cal = imp.getCalibration()  # pixel size (1 x 1 if the image is uncalibrated)
    area = count * cal.pixelWidth * cal.pixelHeight
    return area, count, total / count


def unique_path(folder, filename):
    """Return folder/filename, or folder/<name>_1<ext>, _2, ... if it already exists."""
    name, ext = os.path.splitext(filename)
    path = os.path.join(folder, filename)
    i = 1
    while os.path.exists(path):
        path = os.path.join(folder, "{}_{}{}".format(name, i, ext))
        i += 1
    return path


#-------------------------
# MAIN
#-------------------------
def main():
    # 1. Check if at least one image is open
    ids = WindowManager.getIDList()  # IDs of open images, in the order they were opened
    if not ids:
        IJ.error("No images open.")
        return

    # 2. Collect the open images, once each, keeping the opening order
    images = []
    seen = set()
    for wid in ids:
        imp = WindowManager.getImage(wid)
        if imp is None or imp.getID() in seen:
            continue
        seen.add(imp.getID())
        images.append(imp)

    if not images:
        IJ.error("No suitable images found!")
        return
    n = len(images)  # total number of images to process

    # 3. Ask the user where to save the output table (None if Cancel is pressed)
    output_dir = IJ.getDirectory("Choose a directory to save data")
    if output_dir is None:
        IJ.error("No output directory is selected!")
        return

    # 4. Measurements
    # Create an empty results table, shown as the "Results" window
    # (IJ.saveAs("Results", ...) below saves this window)
    rt = ResultsTable()
    rt.show("Results")

    # Loop over images: measure each one and add one row to the table
    for call_id, imp in enumerate(images, start=1):
        # Progress message in the Log window
        IJ.log("Processing {}/{}: {}".format(call_id, n, imp.getTitle()))

        try:
            area, area_px, mean = nucleus_stats(imp)

            # Add one row: image title, nucleus area, mean nuclear intensity (MFI)
            rt.incrementCounter()
            rt.addValue("Filename", imp.getTitle())
            rt.addValue("Area", area)
            rt.addValue("Area_px", area_px)
            rt.addValue("Mean", mean)

        except Exception as e:
            # Log the error and continue with the next image
            # (the failed image gets no row in the table)
            IJ.log("ERROR in {}: {}".format(imp.getTitle(), e))
            IJ.log(traceback.format_exc())  # comment out if too verbose
            continue

    rt.show("Results")  # refresh the window with the filled table

    # 5. Save the Results table as CSV, without overwriting an existing file
    results_path = unique_path(output_dir, "MFI_nuclei.csv")
    IJ.saveAs("Results", results_path)
    IJ.log("Saved: " + results_path)

    # 6. Clean up: close the saved table and all measured images (without saving)
    close_all_csv_tables()
    close_images(images)


if __name__ == "__main__":
    main()
