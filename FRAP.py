from ij import IJ
from ij.plugin.frame import RoiManager
from ij.measure import ResultsTable
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

def close_image(imp):
    imp.changes = False
    imp.close()

def measure_rois(imp, rois):
    """
    single_ch_imp : ImagePlus (1-channel, may still be Z/T)
    rois          : list of Roi
    out_csv_path  : str
    """    
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
    return rt
    
def main():
    # Load image and process its title
    imp = IJ.getImage()
    if imp is None:
        IJ.error("No image open.")
        return
    img_title = imp.getTitle()
    safe_title = safe_name(img_title)

    # Run ROI manager
    rm = get_rm()
    rois = rm.getRoisAsArray()
    if rois is None or len(rois) == 0:
        IJ.error("No ROIs in ROI Manager.")
        return

    # Ask which channel is your fluorophore + where to save
    output_dir = IJ.getDirectory("Choose a directory to save data")
    if output_dir is None:
        IJ.error("No output directory selected!")
        return

    # Measure Mean internsity over time for the ROIs in ROI manager
    data = measure_rois(imp, rois)

    # Save results in .CSV file
    data_path = os.path.join(output_dir, safe_title + ".csv")
    data.save(data_path)

    # Save ROIs
    zip_path = os.path.join(output_dir, safe_title + "_ROIs.zip")
    rm.runCommand("Save", zip_path)
    rm.reset() # clean ROI manager

    IJ.log("Saved ROI mean intensities and ROIs.zip to: " + output_dir)
    close_image(imp)

main()

