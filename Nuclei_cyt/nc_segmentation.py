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
    gd = GenericDialog("Nuclei and cytoplasm semi-manual segmentation params")
    gd.addMessage("Set parameters.")
    
    lut_options = [
        "Blue",
        "Green",
        "Red",
        "Magenta",
        "Cyan",
        "Yellow",
        "Fire",
        "Grays"
    ]

    # Fields
    gd.addNumericField("DAPI channel (1-based):", 1, 0)
    gd.addNumericField("Measurement channel (1-based):", 2, 0)
    gd.addNumericField("Brightfield channel (1-based):", 3, 0)
    
    # LUT choices
    gd.addChoice("DAPI LUT:", lut_options, "Blue")
    gd.addChoice("Measurement LUT:", lut_options, "Green")
    gd.addChoice("Brightfield LUT:", lut_options, "Grays")
    
    gd.addCheckbox("Apply background subtraction", True)
    gd.addNumericField("Background value (rolling ball radius or constant):", 25, 0)

    gd.showDialog()
    if gd.wasCanceled():
        return None

    params = {}
    params["DAPI_CHANNEL"] = int(gd.getNextNumber())
    params["MEASURE_CHANNEL"] = int(gd.getNextNumber())
    params["BRIGHTFIELD_CHANNEL"] = int(gd.getNextNumber())
    params["DAPI_LUT"] = gd.getNextChoice()
    params["MEASURE_LUT"] = gd.getNextChoice()
    params["BRIGHTFIELD_LUT"] = gd.getNextChoice()
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

def apply_lut(imp, lut_name):
    imp.show()
    imp.setDisplayMode(IJ.GRAYSCALE)

    IJ.selectWindow(imp.getTitle())
    IJ.run(imp, lut_name, "")

    imp.updateAndDraw()

def close_images(imps):
    for im in imps:
        if im is None:
            continue
        im.changes = False
        im.close()
        
def close_image(imp):
    imp.changes = False
    imp.close()

def close_all_csv_tables():
    windows = WindowManager.getAllNonImageWindows()
    if windows is not None:
        for w in windows:
            title = w.getTitle()
            if title.endswith(".csv"):
                w.dispose()
                
def cleanup_iteration():
    rm = RoiManager.getInstance()
    if rm is not None:
        rm.reset()
        rm.close()
        
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
    

def ask_user_to_draw_roi(title, message, roi_name, rm):
    gd = NonBlockingGenericDialog(title)
    gd.addMessage(message)
    gd.showDialog()   # non-blocking UI still works
    if gd.wasCanceled():
        IJ.showMessage("Cancelled. Stopping.")
        return None
    
    # Returns all ROIs currently stored in ROI Manager.
    rois = rm.getRoisAsArray()
    
    # Check if any ROIs were drawn
    if len(rois) == 0 or rois is None:
        IJ.showMessage("No ROI was drawn.")
        return None
    
    # Select one ROI - nucleus or whole-cell, depending on the number of ROIs
    if len(rois) == 1:
        roi = rois[0]
    elif len(rois) == 2:
        roi = rois[1]
    else:
        IJ.showMessage("More than 2 ROIs found. Please draw only one ROI.")
        return None
    
    # Rename the ROI in the ROI Manager to include the user-defined name
    auto_roi_name = roi.getName()
    new_roi_name = roi_name + "_" + auto_roi_name
    roi_index = rm.getCount() - 1 
    rm.rename(roi_index, new_roi_name)  # Rename ROI in ROI Manager
    #print("ROI index:", roi_index, "New ROI name:", new_roi_name)

    return roi

def create_cytoplasm_roi(cell_index, nucleus_index, rm, roi_name="Cytoplasm"):
    # Take 2 ROIs from the ROI Manager
    cell_roi = rm.getRoi(cell_index)
    nucleus_roi = rm.getRoi(nucleus_index)
    
    # Check if ROIs are valid
    if cell_roi is None:
        raise Exception("Cell ROI is None")

    if nucleus_roi is None:
        raise Exception("Nucleus ROI is None")
    
    cytoplasm_roi = ShapeRoi(cell_roi).xor(ShapeRoi(nucleus_roi))

    rm.addRoi(cytoplasm_roi)
    
    roi_index = rm.getCount() - 1
    auto_roi_name = cytoplasm_roi.getName()
    new_roi_name = roi_name + "_" + auto_roi_name
    rm.rename(roi_index, new_roi_name)
    #print("ROI index:", roi_index, "New ROI name:", new_roi_name)

    return cytoplasm_roi

def is_nucleus_inside_cell(nucleus_roi, cell_roi):
    polygon = nucleus_roi.getPolygon()
    for x, y in zip(polygon.xpoints, polygon.ypoints):
        if not cell_roi.contains(x, y):
            return False
    return True
    
def measure_current_channel(imp, rm):
    # Create a new ResultsTable to store measurements
    rt = ResultsTable()
    rois = rm.getRoisAsArray()
    
    # Measure AREA and MEAN for each ROI in the current channel
    for roi in rois:
        roi_name = roi.getName()
        
        imp.setRoi(roi)
        imp.updateAndDraw()
        
        stats = imp.getStatistics(Measurements.AREA | Measurements.MEAN)
        
        # Fill the table with results
        rt.incrementCounter()
        rt.addValue("ROI", roi_name)
        rt.addValue("Area", stats.area)
        rt.addValue("Mean", stats.mean)
        
        # Remove ROI selection
        imp.killRoi()
        
    rt.show("ROI measurements")
    
def ask_continue_same_image(image_title, repeat_id):
    gd = GenericDialog("Continue?")
    gd.addMessage(
        "Finished object %d for image:\n%s" %
        (repeat_id, image_title)
    )

    gd.addChoice(
        "Next action:",
        ["Process another ROI in this image",
         "Go to next image",
         "Stop analysis"],
        "Process another ROI in this image"
    )

    gd.showDialog()

    if gd.wasCanceled():
        return "next_image"

    return gd.getNextChoice()
    
    
def image_processing(imp, p, repeat_id, output_dir="."):
    '''
    This function process semi-manually a single image
    imp - image
    p - parameters
    imp, output_dir, params
    '''
    # Parameteres
    DAPI_CHANNEL = p["DAPI_CHANNEL"] # integer
    MEASURE_CHANNEL = p["MEASURE_CHANNEL"] # integer
    BRIGHTFIELD_CHANNEL = p["BRIGHTFIELD_CHANNEL"] # integer
    dapi_lut = p["DAPI_LUT"] # string
    measure_lut = p["MEASURE_LUT"] # string
    brightfield_lut = p["BRIGHTFIELD_LUT"] # string
    substruct_bg = p["do_bg_subtraction"] # bool
    bg_radius = p["bg_value"] # numeric
    
    try:
        # Processing image title
        img_title = imp.getTitle()
        img_title = img_name_processing(img_title)
        
        # copy original image to avoid changes in the original image
        imp_copy = imp.duplicate()
    
        # Split channels into separate images (C1, C2, ...)
        split_imps = split_channels(imp_copy)
        
        # Select DAPI channel image (used for nuclei segmentation)
        dapi_imp = pick_channel_by_index(split_imps, DAPI_CHANNEL)
        
        # Select the measurement channel image (used for mean intensity measurement)
        meas_original  = pick_channel_by_index(split_imps, MEASURE_CHANNEL)
        
        # Select the brightfield channel image (used for cytoplasmic segmentation)
        brightfield_imp = pick_channel_by_index(split_imps, BRIGHTFIELD_CHANNEL)
    
        # Check splitting
        if dapi_imp is None or meas_original is None or brightfield_imp is None:
            IJ.error("Missing channels for: " + img_title)
            close_images(split_imps)
            return
    
        # Apply LUTs to each channel image
        apply_lut(dapi_imp, dapi_lut)
        apply_lut(meas_original, measure_lut)
        apply_lut(brightfield_imp, brightfield_lut)
    
        # Automatically adjust brightness/contrast for each splitted image (display only)
        for split_img in split_imps:
            split_img.getProcessor().resetMinAndMax()   # reset first
            IJ.run(split_img, "Enhance Contrast", "saturated=2.0")
            split_img.updateAndDraw()
            
        # Make an independent copy of the measurement channel image for background subtraction
        meas_imp = meas_original.duplicate()
        meas_imp.setTitle("Measurement_bg_subtracted")
        
        # --- Background substurction in MEASUREMENT channel ---
        if substruct_bg:
            subtract_background(meas_imp, bg_radius, light_background=False, use_paraboloid=False, do_presmooth=True)
            
        # Run ROI manager
        rm =  ensure_roi_manager(reset=True) # clean roi manager before launch
    
        # User is drawing nucleus ROI on the DAPI channel image: title, message, roi_name, rm
        nucleus_roi = ask_user_to_draw_roi(  
            "Draw nucleus",
            "Draw the nucleus ROI on the image.\n\n"
            "Then click OK.",
            "Nucleus",
            rm
        )
    
        # User is drawing whole-cell ROI on the brightfield channel image
        cell_roi = ask_user_to_draw_roi(
            "Draw whole cell",
            "Draw the whole-cell ROI on the image.\n\n"
            "Then click OK.",
            "Whole_cell",
            rm
        )
    
        if not nucleus_roi or not cell_roi:
            IJ.log("ROI drawing was cancelled or failed. Skipping this image.")
            return
        
        if not is_nucleus_inside_cell(nucleus_roi, cell_roi):
            IJ.log("Warning: Nucleus ROI is not inside the whole-cell ROI!")    
    
        # Create cytoplasm ROI by subtracting nucleus ROI from whole-cell ROI
        cytoplasm_roi = create_cytoplasm_roi(
            cell_index=rm.getCount() - 1,  # last added ROI is whole-cell
            nucleus_index=rm.getCount() - 2,  # second last added ROI is nucleus
            rm=rm,
            roi_name="Cytoplasm"
        )
    
        # Measure AREA and MEAN in the measurement channel
        try:
            measure_current_channel(meas_imp, rm)
        except Exception as e:
            IJ.log("Error during measurement: {}".format(e))
            IJ.log(traceback.format_exc())
            return
    
        # --- SAVE RESULTS ---
        # Save ROIs to a .zip file
        roi_path = os.path.join(output_dir, "{}_{}_rois.zip".format(repeat_id, img_title))
        rm.runCommand("Save", roi_path)
        
        # Save Results table as .csv file
        table_name = "C{}_{}_{}_rois.csv".format(MEASURE_CHANNEL, repeat_id, img_title)
        results_path = os.path.join(output_dir, table_name)
        IJ.saveAs("Results", results_path)
    
    finally:
        # Cleanup: close all split images, close CSV tables, reset ROI manager
        close_images(split_imps)
        close_image(meas_imp)
        close_image(imp_copy)
        
        close_all_csv_tables()
        cleanup_iteration()
   
# --- MAIN FUNCTION ---    
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
        else:
            images.append(imp)
        
    # Check if there are some suitable images
    if not images:
        IJ.error("No suitable images found (only derived windows are open)!")
        return
    
    # Keep only unique images
    unique_images = sorted(list(set(images)))
    n = len(unique_images) # total amount of images to process
    
    # Ask user about the parameters
    params = ask_params_for_image()
    if params is None:
        IJ.error("No parameters provided!")
        return
    
    # Ask user where to save outputs
    output_dir = IJ.getDirectory("Choose a directory to save data")
    if output_dir is None:
        IJ.error("No output directory is selected!")
        return
    
    # ---- Loop: show GUI per image, then process ----
    stop_all = False
    
    for call_id, imp in enumerate(unique_images, start=1):
        repeat_id = 1
        
        while True:
        
            # Make Log message
            msg = "Processing image {}/{}: {}, object {}".format(
            call_id,
            n,
            imp.getTitle(),
            repeat_id
            )
            IJ.log(msg)
        
            try:
                image_processing(imp, params, repeat_id, output_dir)
        
            except Exception as e:
                # log immediately
                IJ.log("ERROR in {}, object {}: {}".format(
                imp.getTitle(),
                repeat_id,
                e
                ))
                IJ.log(traceback.format_exc())  # comment out if too verbose
                break
        
            finally:
                cleanup_iteration()
            
            action = ask_continue_same_image(
            imp.getTitle(),
            repeat_id
            )
            
            if action == "Process another ROI in this image":
                repeat_id += 1
                continue

            elif action == "Go to next image":
                close_image(imp) # close the current multichannel image
                break

            elif action == "Stop analysis":
                stop_all = True
                close_image(imp) # close the current multichannel image
                break
        
        if stop_all:
            break
    
    IJ.showMessage("Finished", "Analysis finished.")


if __name__ == "__main__":
    main()