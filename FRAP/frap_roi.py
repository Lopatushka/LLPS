from ij import IJ, WindowManager
from ij.plugin.frame import RoiManager
from ij.measure import ResultsTable
from ij.process import ImageStatistics
from ij.measure import Measurements
from ij.gui import NonBlockingGenericDialog
import os

def get_rm():
    rm = RoiManager.getInstance() 
    if rm is None:
        rm = RoiManager()
    rm.setVisible(True)
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

    # Keep only image from the channel #2
    ids = WindowManager.getIDList()
    if ids is not None:
        for img_id in ids:
            imp = WindowManager.getImage(img_id)
            if "C=1" not in imp.getTitle():
                close_image(imp)

    # Automatically adjust brightness/contrast (display only)
    imp = IJ.getImage()
    imp.getProcessor().resetMinAndMax()   # reset first
    IJ.run(imp, "Enhance Contrast", "saturated=0.35")
    imp.updateAndDraw()

    # Run ROI manager
    rm = get_rm()
    rois = rm.getRoisAsArray()
    if rois is None or len(rois) == 0:
        gd = NonBlockingGenericDialog("ROI Manager is empty")
        gd.addMessage(
        "Draw ROI(s) on the image, then click 'Add' in ROI Manager.\n"
        "When finished, click OK here to continue."
        )
        gd.showDialog()   # non-blocking UI still works
        if gd.wasCanceled():
            IJ.error("Canceled. Stopping.")
            return
        
    # re-fetch after user interaction
    rois = rm.getRoisAsArray()

    # still empty -> now it's a real stop
    if rois is None or len(rois) == 0:
        IJ.error("Still no ROIs in ROI Manager. Stopping.")
        return

    # Ask which channel is your fluorophore + where to save
    output_dir = IJ.getDirectory("Choose a directory to save data")
    if output_dir is None:
        IJ.error("No output directory selected!")
        return

    # Measure Mean internsity over time for the ROIs in ROI manager
    data = measure_rois(imp,
                        rois,
                        time_factors = [(37, 0.659), (float("inf"), 5)])

    # Save results in .CSV file
    data_path = os.path.join(output_dir, safe_title + ".csv")
    data.save(data_path)

    # Save ROIs
    zip_path = os.path.join(output_dir, safe_title + "_ROIs.zip")
    rm.runCommand("Save", zip_path)
    rm.reset() # clean ROI manager

    #IJ.log("Saved ROI mean intensities and ROIs.zip to: " + output_dir)
    close_image(imp)

main()

