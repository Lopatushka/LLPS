from ij import IJ, WindowManager
from ij.gui import GenericDialog
from ij.plugin.frame import RoiManager
from ij.gui import ShapeRoi
from ij.plugin.filter import BackgroundSubtracter
from ij.measure import Measurements, ResultsTable
from ij.process import ImageStatistics
from ij.gui import NonBlockingGenericDialog
from ij.gui import WaitForUserDialog
from ij.plugin.filter import Analyzer
import os
import csv
import traceback

def ask_params_for_image():
    gd = GenericDialog("Nuclei segmentation params")
    gd.addMessage("Set parameters for nuclei segmentation.")

    # Fields
    gd.addNumericField("DAPI channel (1-based):", 1, 0)
    gd.addNumericField("Measurement channel (1-based):", 2, 0)
    gd.addCheckbox("One nucleus per image", True)
    gd.addCheckbox("Apply background subtraction", True)
    gd.addNumericField("Background value (rolling ball radius or constant):", 25, 0)

    gd.showDialog()
    if gd.wasCanceled():
        return None

    params = {}
    params["DAPI_CHANNEL"] = int(gd.getNextNumber())
    params["MEASURE_CHANNEL"] = int(gd.getNextNumber())
    params["one_roi"] = bool(gd.getNextBoolean())
    params["do_bg_subtraction"] = bool(gd.getNextBoolean())
    params["bg_value"] = float(gd.getNextNumber())

    return params

def img_name_processing(name):
    try:
        if "MP" in name and " - " in name:
            if "Deconvolved" in name:
                name = name.split("-")[0] + "_" + name.split("-")[2]
                name = name.replace(" ", "", 1).replace(",", "").replace(" ", "_")
            else:
                name = name.split("-")[1] # split string
                name = name.replace(" ", "", 1) # delete fist blank in the string
                name = name.replace(" ", "_") # repalce other blanks to underscore
        else:
            name = os.path.splitext(name)[0] # delete extention
        return name
    except Exception as e:
         raise Exception("ERROR in parsing image name")

def ensure_roi_manager(reset=True):
	"""
    Gets the ROI Manager instance.
    Optionally resets it to avoid mixing old ROIs with new ones.
    """
	rm = RoiManager.getInstance()
	if rm is None:
		rm = RoiManager()
	if reset:
		rm.reset()
	return rm

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
    
    return split_imps

def pick_channel_by_index(split_imps, one_based_index):
	"""
    Picks a channel ImagePlus from split_imps using 1-based indexing.
    Example: one_based_index=1 -> C1
    """
	idx = int(one_based_index) - 1
	if idx < 0 or idx >= len(split_imps):
		return None
	return split_imps[idx]

def close_images(imps):
    for im in imps:
        if im is None:
            continue
        im.changes = False
        im.close()

def subtract_background(imp, radius, light_background=False, use_paraboloid=False, do_presmooth=True):
    radius = float(radius)
    ip = imp.getProcessor()  # ImageProcessor of current slice
    BackgroundSubtracter().rollingBallBackground(
        ip,
        radius,
        False,
        bool(light_background),
        bool(use_paraboloid),
        bool(do_presmooth),
        False
    )
    imp.updateAndDraw()
    
def semi_manual_img_process(imp, output_dir, p):
    '''
    This function process semi-manually a single image
    imp - image
    p - parameters
    imp, output_dir, params
    '''
    # Parameteres
    DAPI_CHANNEL = p["DAPI_CHANNEL"]
    MEASURE_CHANNEL = p["MEASURE_CHANNEL"]
    one_roi = p["one_roi"]
    substruct_bg = p["do_bg_subtraction"] # bool
    bg_radius = p["bg_value"] # numeric

    # Processing image title
    img_title = imp.getTitle()
    img_title = img_name_processing(img_title)

    # Split channels into separate images (C1, C2, ...)
    split_imps = split_channels(imp)
    
    # Select DAPI channel image (used for nuclei segmentation)
    dapi_imp = pick_channel_by_index(split_imps, DAPI_CHANNEL)

    # Select the measurement channel image (used for mean intensity measurement)
    meas_imp = pick_channel_by_index(split_imps, MEASURE_CHANNEL)

    # Check splitting
    if dapi_imp is None or meas_imp is None:
        IJ.error("Missing channels for: " + img_title)
        close_images(split_imps)
        return
    
    # Automatically adjust brightness/contrast for each splitted image (display only)
    for split_img in split_imps:
        split_img.getProcessor().resetMinAndMax()   # reset first
        IJ.run(split_img, "Enhance Contrast", "saturated=0.35")
        split_img.updateAndDraw()
    
    # --- Background substurction in MEASUREMENT channel ---
    if substruct_bg:
        subtract_background(meas_imp, bg_radius, light_background=False, use_paraboloid=False, do_presmooth=True)

    # Run ROI manager
    rm =  ensure_roi_manager(reset=True) # clean roi manager before launch
    rois = rm.getRoisAsArray() # list of ROIs in roi manager

    # Bring this image above all others
    dapi_imp.getWindow().toFront()

    # WHILE Loop to fill Roi manager
    while len(rois) == 0:
        gd = NonBlockingGenericDialog("ROI Manager is empty")
        gd.addMessage(
        "Draw ROI(s) on the image, then click 'Add' in ROI Manager.\n"
        "When finished, click OK here to continue."
        )
        gd.showDialog()   # non-blocking UI still works

        # If user press Cancel
        if gd.wasCanceled():
            IJ.error("Cancelled. Stopping.")
            break

        # re-fetch after user interaction in while loop
        rois = rm.getRoisAsArray()

        # Check how many ROIs we have if the user wants 1 ROI per image
        if one_roi:
            if len(rois) != 1:
                WaitForUserDialog(
                "ROI Warning",
                "Choose exactly one ROI!").show()
                # Clean ROI manager
                rm.reset()
                # re-fetch after user interaction in while loop
                rois = rm.getRoisAsArray()  
                continue

    # If ROI manager is empty, stop the program.
    if len(rois) == 0:
        return

    # --- Measuremtemts of ROIs ---
    # Results table
    rt = ResultsTable()

    # Set ROIs at the MEASUREMENT images
    for i, roi in enumerate(rois):
        roi_name = roi.getName()
        if roi_name is None or roi_name.strip() == "":
            roi_name = "ROI_%02d" % (i + 1)

        # Duplicate the MEASUREMENT channel and show it
        meas_imp_work = meas_imp.duplicate()
        meas_imp_work.setTitle("MEAS_work")
        meas_imp_work.show()

        # Clear everything else outside desired ROI in the copied image
        meas_imp_work.setRoi(roi)
        IJ.run(meas_imp_work, "Clear Outside", "")

        # Make measurememts on the WORK image
        stats = meas_imp_work.getStatistics(
        Measurements.AREA | Measurements.MEAN
        )

        # Fill the table with results
        rt.incrementCounter()
        rt.addValue("ROI", roi_name)
        rt.addValue("Area", stats.area)
        rt.addValue("Mean", stats.mean)
        
        # Remove ROI selection from WORK image
        meas_imp_work.killRoi()

        # Close WORK image w/o saving
        meas_imp_work.changes = False
        meas_imp_work.close()

    # Show results
    rt.show("ROI measurements")

    # --- SAVE DATA ---
    # Save MEASUREMENT channel
    MEASURE_CHANNEL_name = "C{}_{}.tif".format(MEASURE_CHANNEL, img_title)
    MEASURE_CHANNEL_path = os.path.join(output_dir, MEASURE_CHANNEL_name)
    #meas_imp.show()
    IJ.save(meas_imp, MEASURE_CHANNEL_path)

    # Close splitted images
    close_images(split_imps)

    return rt

def _append_rt(final_rt, small_rt, image_name=None):
    for r in range(small_rt.size()):
        final_rt.incrementCounter()

        if image_name is not None:
            final_rt.addValue("Image", image_name)

        headings = small_rt.getHeadings()
        for h in headings:
            value = small_rt.getValue(h, r)
            final_rt.addValue(h, value)

def main():
    # Check if at least one image is opened
    ids = WindowManager.getIDList()
    if not ids:
        IJ.error("No images open.")
        return
    
    # Opened images checking and filtration
    images = [] # store images in the list
    for wid in ids:
        imp = WindowManager.getImage(wid)
        if imp is None:
            continue
        title = imp.getTitle()

        # Skip typical derived images (adjust if needed)
        if (title.startswith("C") and "-" in title) or title in ["DAPI_work", "Nuclei_mask_particles_only"]:
            continue         
        images.append(imp)
    
    # Check if there are some suitable images after filtration
    if not images:
        IJ.error("No suitable images found (only derived windows are open)!")
        return
    
    # Keep only unique images
    unique_images = list(set(images))
    n = len(unique_images) # total amount of images to process

    # Ask user about the parameters
    params = ask_params_for_image()
    if params is None:
        IJ.error("No parameters provided!")
        return

    # Ask user where to save outputs
    output_dir = IJ.getDirectory("Choose a directory to save data")
    if output_dir is None:
        IJ.error("No output directory selected!")
        return
    
    # Collect all errors here
    errors = []  

    # ---- Loop: show GUI per image, then process ----
    for call_id, imp in enumerate(unique_images, start=1):
        # Make Log message
        msg = "Processing {}/{}: {}".format(call_id, n, imp.getTitle())
        IJ.log(msg)

        try:
            semi_manual_img_process(imp, output_dir, params)

        except Exception as e:
            # log immediately
            IJ.log("ERROR in {}: {}".format(imp.getTitle(), e))
            IJ.log(traceback.format_exc())  # comment out if too verbose
            continue


# Run program
main()
