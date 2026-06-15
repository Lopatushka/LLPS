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
    
def image_processing(imp, output_dir, p):
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



def cleanup_iteration():
    rm = RoiManager.getInstance()
    if rm is not None:
        rm.reset()
        rm.close()

def draw_nuclei(rm):
    IJ.setTool("freehand")

    WaitForUserDialog(
        "Nuclei Segmentation",
        "Draw nuclei ROIs.\n"
        "Press 't' after each ROI to add it to ROI Manager.\n"
        "Click OK when finished."
    ).show()

    n_nuclei = rm.getCount()

    for i in range(n_nuclei):
        rm.select(i)
        rm.rename("Nucleus_%d" % (i + 1))

    return n_nuclei

def draw_cells(rm, n_nuclei):
    WaitForUserDialog(
        "Cell Segmentation",
        "Draw whole-cell ROIs.\n"
        "Press 't' after each ROI.\n"
        "Click OK when finished."
    ).show()

    total_rois = rm.getCount()
    n_cells = total_rois - n_nuclei

    for i in range(n_cells):
        rm.select(n_nuclei + i)
        rm.rename("Cell_%d" % (i + 1))

    return n_cells

def create_mask(width, height, title):
    return IJ.createImage(title, "8-bit black", width, height, 1)


def fill_mask(mask, rois):
    for roi in rois:
        mask.setRoi(roi)
        IJ.run(mask, "Fill", "slice")
        
def build_masks(rm, n_nuclei, n_cells, width, height):
    nuclei_mask = create_mask(width, height, "Nuclei_mask")
    cell_mask = create_mask(width, height, "Cell_mask")

    nuclei_rois = [rm.getRoi(i) for i in range(n_nuclei)]
    cell_rois = [rm.getRoi(n_nuclei + i) for i in range(n_cells)]

    fill_mask(nuclei_mask, nuclei_rois)
    fill_mask(cell_mask, cell_rois)

    return nuclei_mask, cell_mask


def create_cytoplasm_mask(cell_mask, nuclei_mask):
    cyto_mask = cell_mask.duplicate()
    cyto_mask.setTitle("Cytoplasm_mask")

    IJ.run(
        cyto_mask,
        "Image Calculator...",
        "image1=Cytoplasm_mask operation=Subtract image2=Nuclei_mask create"
    )

    result = WindowManager.getImage("Result of Cytoplasm_mask")

    if result:
        result.setTitle("Cytoplasm_mask")

    return result

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
        
    # Check if there are some suitable images
    if not images:
        IJ.error("No suitable images found (only derived windows are open)!")
        return
    
    # Keep only unique images
    unique_images = list(set(images))
    n = len(unique_images) # total amount of images to process
    
    # Ask user where to save outputs
    output_dir = IJ.getDirectory("Choose a directory to save data")
    if output_dir is None:
        IJ.error("No output directory is selected!")
        return
    
    # ---- Loop: show GUI per image, then process ----
    for call_id, imp in enumerate(unique_images, start=1):
        # Make Log message
        msg = "Processing {}/{}: {}".format(call_id, n, imp.getTitle())
        IJ.log(msg)



    rm = get_roi_manager()

    n_nuclei = draw_nuclei(rm)
    n_cells = draw_cells(rm, n_nuclei)

    nuclei_mask, cell_mask = build_masks(
        rm,
        n_nuclei,
        n_cells,
        imp.getWidth(),
        imp.getHeight()
    )

    cyto_mask = create_cytoplasm_mask(
        cell_mask,
        nuclei_mask
    )

    results_table = measure_rois(
        imp,
        rm,
        n_nuclei,
        n_cells
    )

    if output_dir:
        save_results(
            output_dir,
            imp.getTitle(),
            nuclei_mask,
            cell_mask,
            cyto_mask,
            rm,
            results_table
        )

    IJ.showMessage(
        "Finished",
        "Segmentation completed successfully."
    )


if __name__ == "__main__":
    main()