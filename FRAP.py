from ij import IJ
from ij.plugin import ChannelSplitter
from ij.plugin.frame import RoiManager
from ij.measure import ResultsTable
from ij.gui import GenericDialog
import os

def get_rm():
    rm = RoiManager.getInstance()
    if rm is None:
        rm = RoiManager()
    return rm

def safe_name(s):
    return "".join([c if c.isalnum() or c in "._- " else "_" for c in s]).strip()

def measure_rois_over_time(single_ch_imp, rois, out_csv_path):
    """
    single_ch_imp : ImagePlus (1-channel, may still be Z/T)
    rois          : list of Roi
    out_csv_path  : str
    """
    nT = single_ch_imp.getNFrames()
    nZ = single_ch_imp.getNSlices()

    rt = ResultsTable()

    # Iterate timepoints (1-based in ImageJ)
    for t in range(1, nT + 1):
        # If Z exists, measure on the current Z (default slice) OR do a projection first.
        # Here: measure on the current Z slice (usually Z=1). Adjust below if you want something else.
        z = 1
        single_ch_imp.setPosition(1, z, t)  # (channel=1, slice=z, frame=t)

        ip = single_ch_imp.getProcessor()

        for i, roi in enumerate(rois):
            single_ch_imp.setRoi(roi)
            stats = ip.getStatistics()  # respects current ROI

            # ROI name (use existing name if present)
            roi_name = roi.getName()
            if roi_name is None or roi_name.strip() == "":
                roi_name = "ROI_%02d" % (i + 1)

            rt.incrementCounter()
            rt.addValue("timepoint", t)
            rt.addValue("roi", roi_name)
            rt.addValue("mean", stats.mean)

    single_ch_imp.killRoi()
    rt.save(out_csv_path)

def main():
    imp = IJ.getImage()
    if imp is None:
        IJ.error("No image open.")
        return

    rm = get_rm()
    rois = rm.getRoisAsArray()
    if rois is None or len(rois) == 0:
        IJ.error("No ROIs in ROI Manager.")
        return

    # Ask which channel is your fluorophore + where to save
    #gd = GenericDialog("Measure ROI mean intensity over time")
    #gd.addNumericField("Fluorophore channel (1-based):", 2, 0)  # change default if needed
    #gd.addStringField("Output CSV path:", os.path.join(IJ.getDirectory("home"), safe_name(imp.getTitle()) + "_roi_means.csv"), 60)
    #gd.showDialog()
    #if gd.wasCanceled():
        #return

    #ch_index = int(gd.getNextNumber())
    #out_csv = gd.getNextString()

    # Split channels
    #ch_imps = ChannelSplitter.split(imp)
    #if ch_index < 1 or ch_index > len(ch_imps):
        #IJ.error("Channel index out of range. Image has %d channel(s)." % len(ch_imps))
        #return

    #fluor_imp = ch_imps[ch_index - 1]
    #fluor_imp.setTitle(safe_name(imp.getTitle()) + "_ch%d" % ch_index)

    # Close other channels to avoid clutter (optional)
    #for idx, ci in enumerate(ch_imps):
        #if idx != (ch_index - 1):
            #ci.close()

    # Measure over time and save CSV
    #measure_rois_over_time(fluor_imp, rois, out_csv)

    #IJ.log("Saved ROI mean intensities to: " + out_csv)
    #IJ.showMessage("Done", "Saved:\n" + out_csv)

main()

