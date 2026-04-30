from ij import IJ, WindowManager
from ij.gui import GenericDialog
from ij.plugin.frame import RoiManager
from ij.measure import Measurements, ResultsTable
from ij.plugin.filter import ParticleAnalyzer
from ij.plugin.filter import BackgroundSubtracter
from ij.process import AutoThresholder
from ij.io import RoiEncoder
from ij.gui import NonBlockingGenericDialog
#from ij.gui import WaitForUserDialog
import os
import csv
import traceback

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
    
def semi_manual_img_process(imp):
    '''
    This function process semi-manually a single image
    imp - image
    p - parameters
    '''
    # Processing image title
    img_title = imp.getTitle()
    img_title = img_name_processing(img_title)

    # Split channels into separate images (C1, C2, ...)
    split_imps = split_channels(imp)

    # Automatically adjust brightness/contrast for each splitted image (display only)
    for split_img in split_imps:
        split_img.getProcessor().resetMinAndMax()   # reset first
        IJ.run(split_img, "Enhance Contrast", "saturated=0.35")
        split_img.updateAndDraw()

    # Run ROI manager
    rm =  ensure_roi_manager(reset=True) # clean roi manager before launch
    rois = rm.getRoisAsArray() # list of ROIs in roi manager

    # WHILE Loop to fill Roi manager
    while len(rois) == 0:
        gd = NonBlockingGenericDialog("ROI Manager is empty")
        gd.addMessage(
        "Draw ROI(s) on the image, then click 'Add' in ROI Manager.\n"
        "When finished, click OK here to continue."
        )
        gd.showDialog()   # non-blocking UI still works

        # re-fetch after user interaction in while loop
        rois = rm.getRoisAsArray()

        # If user press Cancel
        if gd.wasCanceled():
            IJ.error("Canceled. Stopping.")
            break
            
    # Re-fetch after user interaction
    #rois = rm.getRoisAsArray()

    # If user press cancell stop the program
    if len(rois) == 0:
        return

        

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
            semi_manual_img_process(imp)

        except Exception as e:
            # log immediately
            IJ.log("ERROR in {}: {}".format(imp.getTitle(), e))
            IJ.log(traceback.format_exc())  # comment out if too verbose
            continue

# Run program
main()
