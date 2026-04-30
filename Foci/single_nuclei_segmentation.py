from ij import IJ, WindowManager
from ij.gui import GenericDialog
from ij.plugin.frame import RoiManager
from ij.measure import Measurements, ResultsTable
from ij.plugin.filter import ParticleAnalyzer
from ij.plugin.filter import BackgroundSubtracter
from ij.process import AutoThresholder
from ij.io import RoiEncoder
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
    
def semi_manual_img_process(imp):
    '''
    This function process semi-manually a single image
    imp - image
    p - parameters
    '''
    # Processing image title
    img_title = imp.getTitle()
    img_title = img_name_processing(img_title)
    
    print(img_title)

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
    
    errors = []  # collect all errors here

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
