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

def measure_rois(imp, rois, time_factors = None):
    """
    Measure mean intensity of multiple ROIs over time.

    Parameters
    ----------
    imp : ImagePlus
        A single-channel ImagePlus object (can be a time series stack).
        Assumes channel=1 and Z=1 are used for measurement.

    rois : list of Roi
        List of ROI objects to measure.

    time_factors : list of tuples or None
        Defines variable time intervals between frames.

        Each tuple must be:
            (max_frame_number, frame_interval_seconds)

        Meaning:
        - max_frame_number : int or float("inf")
            Upper frame limit (1-based indexing).
        - frame_interval_seconds : float
            Time interval in seconds between consecutive frames
            up to that frame limit.

        Example:
            [(50, 0.133), (float("inf"), 5)]

        Interpretation:
            Frames 1–49   → interval = 0.133 sec
            Frames ≥50    → interval = 5 sec

        The first tuple where current_frame < max_frame_number
        determines the interval used.

        If None:
            Frame number is used directly as "Time".
    """
    
    nframes = imp.getNFrames()
    #nZ = single_ch_imp.getNSlices()
    rt = ResultsTable()

    # Loop over time
    time = 0
    for n in range(1, nframes + 1):
        # If Z exists, measure on the current Z (default slice) OR do a projection first.
        # Here: measure on the current Z slice (usually Z=1). Adjust below if you want something else.
        z = 1
        imp.setPosition(1, z, n)  # (channel=1, slice=z, frame=t)
        rt.incrementCounter()
        if time_factors:
            for nframe, sec in time_factors:
                if n < nframe:
                    time += sec
                    rt.addValue("Time", time)
                    break
                else:
                    continue       
        else:
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
    data = measure_rois(imp,
                        rois,
                        time_factors = [(50, 0.133), (float("inf"), 5)])

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

