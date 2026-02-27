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
    return s.split(".")[0].strip()

def close_images(imps):
    for im in imps:
        if im is None:
            continue
        im.changes = False
        im.close()

def measure_rois(imp):
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
    
    nT = imp.getNFrames()
    #nZ = single_ch_imp.getNSlices()
    rt = ResultsTable()

    # Loop over time
    for t in range(1, nT + 1):
        # If Z exists, measure on the current Z (default slice) OR do a projection first.
        # Here: measure on the current Z slice (usually Z=1). Adjust below if you want something else.
        z = 1
        imp.setPosition(1, z, t)  # (channel=1, slice=z, frame=t)
        rt.incrementCounter()
        rt.addValue("Time", t)

        for i, roi in enumerate(rois):
            roi_name = roi.getName()
            if roi_name is None or roi_name.strip() == "":
                roi_name = "ROI_%02d" % (i + 1)

            imp.setRoi(roi)

            stats = ImageStatistics.getStatistics(
            imp.getProcessor(),
            Measurements.MEAN,
            imp.getCalibration()
            )

            rt.addValue(roi_name, stats.mean)

    imp.killRoi()
    #rt.show("My Results")
    #rt.save(out_csv_path)
    return rt
    
def main():
    # Load image and process its title
    imp = IJ.getImage()
    if imp is None:
        IJ.error("No image open.")
        return
    img_title = imp.getTitle()
    safe_title = safe_name(img_title)
    print(safe_title)

    # Ask which channel is your fluorophore + where to save
    output_dir = IJ.getDirectory("Choose a directory to save data")
    if output_dir is None:
        IJ.error("No output directory selected!")
        return

    # Measure over time and save CSV
    data = measure_rois(imp)
    data_path = os.path.join(output_dir, safe_title + ".csv")
    print(data_path)
    #data.save(data_path)


    #IJ.log("Saved ROI mean intensities to: " + output_dir)
    #close_images(imp)

main()

