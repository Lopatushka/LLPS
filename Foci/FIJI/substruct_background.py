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
import traceback

def ask_params_for_image():
    gd = GenericDialog("Substruct Background Parameters")
    gd.addMessage("Set parameter for background subtraction.")

    # Fields
    gd.addNumericField("Background value (rolling ball radius or constant):", 15, 0)

    gd.showDialog()
    if gd.wasCanceled():
        return None

    params = {}
    params["bg_value"] = float(gd.getNextNumber())

    return params

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
    
def close_images(imps):
    for im in imps:
        if im is None:
            continue
        im.changes = False
        im.close()
        
#-------------------------
# MAIN
#-------------------------
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
        IJ.error("No output directory is selected!")
        return

    # ---- Loop: show GUI per image, then process ----
    for call_id, imp in enumerate(unique_images, start=1):
        # Make Log message
        msg = "Processing {}/{}: {}".format(call_id, n, imp.getTitle())
        IJ.log(msg)
    
        try:
            # Subtract background
            subtract_background(imp = imp, radius = params["bg_value"])
            
            # Save the processed image
            output_path = os.path.join(output_dir, imp.getTitle())
            IJ.saveAs(imp, "Tiff", output_path)
            IJ.log("Saved processed image to: {}".format(output_path))
            
        except Exception as e:
            IJ.error("Error processing image {}: {}".format(imp.getTitle(), str(e)))
            traceback.print_exc()
            
if __name__ == "__main__":
    main()