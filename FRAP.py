from ij import IJ, WindowManager
from ij.plugin import ChannelSplitter
from ij.plugin.frame import RoiManager
from ij.measure import ResultsTable
from ij.gui import GenericDialog
from ij.process import ImageStatistics
from ij.measure import Measurements
import os

def get_rm():
    rm = RoiManager.getInstance()
    if rm is None:
        rm = RoiManager()
    return rm

def safe_name(s):
    return "".join([c if c.isalnum() or c in "._- " else "_" for c in s]).strip()

def close_images(keep_imp = None):
    """
    Close all open images except keep_imp (ImagePlus).
    """
    open_ids = WindowManager.getIDList()
    if open_ids is None:
        return

    for img_id in open_ids:
        imp = WindowManager.getImage(img_id)
        if imp is not None and imp != keep_imp:
            imp.close()

def measure_rois_wide(imp):
    """
    single_ch_imp : ImagePlus (1-channel, may still be Z/T)
    rois          : list of Roi
    out_csv_path  : str
    """
    # Run ROI manager
    rm = get_rm()
    rois = rm.getRoisAsArray()
    if rois is None or len(rois) == 0:
        IJ.error("No ROIs in ROI Manager.")
        return
    
    nT = single_ch_imp.getNFrames()
    #nZ = single_ch_imp.getNSlices()
    rt = ResultsTable()

    # Loop over time
    for t in range(1, nT + 1):
        # If Z exists, measure on the current Z (default slice) OR do a projection first.
        # Here: measure on the current Z slice (usually Z=1). Adjust below if you want something else.
        z = 1
        single_ch_imp.setPosition(channel, z, t)  # (channel=1, slice=z, frame=t)

        ip = single_ch_imp.getProcessor()

        for i, roi in enumerate(rois):
            single_ch_imp.setRoi(roi)
            stats = ImageStatistics.getStatistics(
            single_ch_imp.getProcessor(),
            Measurements.MEAN,
            single_ch_imp.getCalibration()
            )

            mean_val = stats.mean

            # ROI name (use existing name if present)
            roi_name = roi.getName()
            if roi_name is None or roi_name.strip() == "":
                roi_name = "ROI_%02d" % (i + 1)

            rt.incrementCounter()
            rt.addValue("timepoint", t)
            rt.addValue("roi", roi_name)
            rt.addValue("mean", mean_val)

    single_ch_imp.killRoi()
    rt.show("My Results")
    #rt.save(out_csv_path)

def main():
    # Load image and process its title
    imp = IJ.getImage()
    if imp is None:
        IJ.error("No image open.")
        return
    img_title = imp.getTitle()

    # Ask which channel is your fluorophore + where to save
    output_dir = IJ.getDirectory("Choose a directory to save data")
    if output_dir is None:
        IJ.error("No output directory selected!")
        return

    # Measure over time and save CSV
    measure_rois_wide(imp)

    #IJ.log("Saved ROI mean intensities to: " + output_dir)

main()

