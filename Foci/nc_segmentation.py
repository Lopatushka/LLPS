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

    # Fields
    gd.addNumericField("DAPI channel (1-based):", 1, 0)
    gd.addNumericField("Measurement channel (1-based):", 2, 0)
    gd.addNumericField("Brightfield channel (1-based):", 3, 0)
    gd.addCheckbox("One nucleus per image", False)
    gd.addCheckbox("Apply background subtraction", True)
    gd.addNumericField("Background value (rolling ball radius or constant):", 25, 0)

    gd.showDialog()
    if gd.wasCanceled():
        return None

    params = {}
    params["DAPI_CHANNEL"] = int(gd.getNextNumber())
    params["MEASURE_CHANNEL"] = int(gd.getNextNumber())
    params["BRIGHTFIELD_CHANNEL"] = int(gd.getNextNumber())
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
    

def ask_user_to_draw_roi(title, message, roi_name, rm):
    gd = NonBlockingGenericDialog(title)
    gd.addMessage(
        "Draw ROI on the image, then click 'Add' in ROI Manager.\n"
        "When finished, click OK here to continue."
        )
    gd.showDialog()   # non-blocking UI still works
    if gd.wasCanceled():
        IJ.showMessage("Cancelled. Stopping.")
        return None
    
    roi = rm.getRoisAsArray() #returns all ROIs currently stored in ROI Manager.

    if roi is None:
        IJ.showMessage("No ROI was drawn.")
        return None

    roi_index = rm.getCount() - 1 
    #rm.rename(roi_index,  roi_name)
    print("ROI index:", roi_index, "ROI name:", roi_name)

    return roi

def create_cytoplasm_roi(nucleus_roi, cell_roi, rm, roi_name="Cytoplasm"):
    # XOR works if nucleus is completely inside the whole-cell ROI.  
    cytoplasm_roi = ShapeRoi(cell_roi).xor(ShapeRoi(nucleus_roi))

    rm.addRoi(cytoplasm_roi)
    roi_index = rm.getCount() - 1 
    #rm.rename(roi_index, roi_name)
    print("ROI index:", roi_index, "ROI name:", roi_name)

    return cytoplasm_roi

def delete_whole_cell_roi(rm, name="Whole_cell"):
    for i in range(rm.getCount()):
        name = rm.getName(i)
        if name == name:
            rm.select(i)
            rm.runCommand("Delete")
            
def measure_current_channel(imp, roi):
    IJ.run("Set Measurements...", "area mean display redirect=None decimal=3")
    imp.setRoi(roi)
    IJ.run(imp, "Measure")
    
def image_processing(imp, p):
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
    one_roi = p["one_roi"] # bool
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

    # Select the brightfield channel image (used for cytoplasmic segmentation)
    brightfield_imp = pick_channel_by_index(split_imps, BRIGHTFIELD_CHANNEL)
    
    # Check splitting
    if dapi_imp is None or meas_imp is None or brightfield_imp is None:
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
    
    # User is drawing nucleus ROI on the DAPI channel image
    # WHILE Loop to fill Roi manager.haha
    while len(rois) == 0:
        nucleus_roi = ask_user_to_draw_roi(  # title, message, roi_name, rm
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
    # Check if the user drew a whole-cell ROI
    if cell_roi is None:
        return
    
    # Create cytoplasm ROI by subtracting nucleus ROI from whole-cell ROI
    cytoplasm_roi = create_cytoplasm_roi(
        nucleus_roi,
        cell_roi,
        rm
    )
    
    # Delete the whole-cell ROI from the ROI Manager, leaving only nucleus and cytoplasm ROIs
    #delete_whole_cell_roi(rm)
    
    # Measure area and mean intensity in the measurement channel for the cytoplasm ROI
    #measure_current_channel(
        #imp,
        #cytoplasm_roi
    #)
    


def cleanup_iteration():
    rm = RoiManager.getInstance()
    if rm is not None:
        rm.reset()
        rm.close()


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
    unique_images = list(set(images))
    n = len(unique_images) # total amount of images to process
    
    # Ask user about the parameters
    params = ask_params_for_image()
    if params is None:
        IJ.error("No parameters provided!")
        return
    
    # Ask user where to save outputs
    #output_dir = IJ.getDirectory("Choose a directory to save data")
    #if output_dir is None:
        #IJ.error("No output directory is selected!")
        #return
    
    # ---- Loop: show GUI per image, then process ----
    for call_id, imp in enumerate(unique_images, start=1):
        # Make Log message
        msg = "Processing {}/{}: {}".format(call_id, n, imp.getTitle())
        IJ.log(msg)
        
        try:
            image_processing(imp, params)
        
        except Exception as e:
            # log immediately
            IJ.log("ERROR in {}: {}".format(imp.getTitle(), e))
            IJ.log(traceback.format_exc())  # comment out if too verbose
            continue
        
        finally:
            #cleanup_iteration()
            IJ.showMessage("Finished.")


if __name__ == "__main__":
    main()