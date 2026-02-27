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

def pick_channel_by_index(split_imps, one_based_index):
	"""
    Picks a channel ImagePlus from split_imps using 1-based indexing.
    Example: one_based_index=1 -> C1
    """
	idx = int(one_based_index) - 1
	if idx < 0 or idx >= len(split_imps):
		return None
	return split_imps[idx]

def split_channels(imp):
    """
    Runs ImageJ command 'Split Channels' on the input image.
    Returns a list of split channel ImagePlus objects that belong to the original image.
    The list is sorted as [C1, C2, C3, ...].
    """
    orig_title = imp.getTitle()

    # IDs before splitting
    before = set(WindowManager.getIDList() or [])
    
    # Split channels: creates new windows like "C1-<orig_title>", "C2-<orig_title>", ...
    IJ.run(imp, "Split Channels", "")

    # IDs after splitting
    after = set(WindowManager.getIDList() or [])
    new_ids = list(after - before)
    
    # Get all currently opened image window IDs
    ids = WindowManager.getIDList()
    if not ids:
        IJ.error("No windows after Split Channels.")
        raise SystemExit

    split_imps = []
    for wid in new_ids:
        wimp = WindowManager.getImage(wid)
        if wimp is None:
            continue
        title = wimp.getTitle()
        # Keep only windows that look like split channels of THIS image
        if title.startswith("C") and "-" in title and (orig_title in title):
            split_imps.append(wimp)

    if len(split_imps) == 0:
        IJ.error("Could not find split channel images. Make sure your image is multichannel/composite.")
        raise SystemExit

    # Sort by channel number: C1, C2, C3...
    def chan_index(t):
        # expects "C2-..." -> 2
        try:
            return int(t.split("-")[0][1:])
        except:
            return 999
    split_imps.sort(key=lambda im: chan_index(im.getTitle()))

    return split_imps

def measure_rois_over_time(single_ch_imp, rois, out_csv_path):
    """
    single_ch_imp : ImagePlus (1-channel, may still be Z/T)
    rois          : list of Roi
    out_csv_path  : str
    """
    nT = single_ch_imp.getNFrames()
    #nZ = single_ch_imp.getNSlices()

    rt = ResultsTable()

    # Iterate timepoints (1-based in ImageJ)
    for t in range(1, nT + 1):
        # If Z exists, measure on the current Z (default slice) OR do a projection first.
        # Here: measure on the current Z slice (usually Z=1). Adjust below if you want something else.
        #z = 1
        #single_ch_imp.setPosition(1, z, t)  # (channel=1, slice=z, frame=t)

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
    img_title = safe_name(img_title)

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

    # Ask about the MEASUREMENT channek number
    gd = GenericDialog("Measure ROI mean intensity over time")
    gd.addNumericField("Fluorophore channel (1-based):", 2, 0)  # change default if needed
    gd.showDialog()
    if gd.wasCanceled():
        return

    ch_index = int(gd.getNextNumber())

    # Split channels and keep MEASURUMENT channel
    split_imps = split_channels(imp)
    if ch_index < 1 or ch_index > len(split_imps):
        IJ.error("Channel index out of range. Image has %d channel(s)." % len(split_imps))
        return
    
    meas_imp = pick_channel_by_index(split_imps, ch_index)
    if meas_imp is None:
        IJ.error("Missing channels for: " + img_title)
        close_images(keep_imp = None)
        return

    # Close other channels to avoid clutter (optional)
    close_images(keep_imp = meas_imp)

    # Measure over time and save CSV
    measure_rois_over_time(meas_imp, rois, output_dir)

    #IJ.log("Saved ROI mean intensities to: " + output_dir)

main()

