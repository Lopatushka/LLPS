from ij import IJ, WindowManager
from ij.gui import GenericDialog
from ij.plugin.filter import BackgroundSubtracter
import os
import traceback

def ask_params_for_image():
    gd = GenericDialog("Subtract Background Parameters")
    gd.addMessage("Set parameter for background subtraction.")

    # Fields
    gd.addNumericField("Rolling ball radius (pixels):", 15, 0)

    gd.showDialog()
    if gd.wasCanceled():
        return None

    radius = float(gd.getNextNumber())
    if radius <= 0:
        IJ.error("Rolling ball radius must be greater than 0.")
        return None

    return {"bg_value": radius}

def subtract_background(imp, radius, light_background=False, use_paraboloid=False, do_presmooth=True):
    """Rolling-ball background subtraction on every slice of the image
    (all Z-slices, channels and time points), not only the displayed one."""
    radius = float(radius)
    stack = imp.getStack()
    bs = BackgroundSubtracter()
    for n in range(1, stack.getSize() + 1):
        ip = stack.getProcessor(n)
        bs.rollingBallBackground(
            ip,
            radius,
            False,                  # createBackground: subtract, don't return the background
            bool(light_background),
            bool(use_paraboloid),
            bool(do_presmooth),
            False                   # correctCorners (only used with paraboloid)
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
    images = [] # store images in the list, in the order they were opened
    for wid in ids:
        imp = WindowManager.getImage(wid)
        if imp is None:
            continue
        # Skip images that were already background-subtracted, to avoid subtracting twice
        if "_bg_subtracted" in imp.getTitle():
            continue
        images.append(imp)

    # Check if there are some suitable images after filtration
    if not images:
        IJ.error("No suitable images found (only already background-subtracted images are open)!")
        return

    n = len(images) # total amount of images to process

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

    # ---- Loop over images ----
    done = []   # images saved successfully; closed at the end
    failed = 0  # images with errors stay open so you can check them
    for call_id, imp in enumerate(images, start=1):
        title = imp.getTitle()
        IJ.log("Processing {}/{}: {}".format(call_id, n, title))

        try:
            # Subtract background
            subtract_background(imp=imp, radius=params["bg_value"])
            IJ.log("Rolling ball raidus is {}".format(params["bg_value"]))

            # Save the processed image (works for any extension, or none)
            name = os.path.splitext(title)[0] + "_bg_subtracted.tif"
            output_path = os.path.join(output_dir, name)
            IJ.saveAs(imp, "Tiff", output_path)
            IJ.log("Saved processed image to: {}".format(output_path))
            done.append(imp)

        except Exception as e:
            # Log instead of a popup, so the batch keeps running
            failed += 1
            IJ.log("ERROR in {}: {}".format(title, e))
            IJ.log(traceback.format_exc())

    close_images(done)
    IJ.log("Finished: {} saved, {} failed.".format(len(done), failed))

if __name__ == "__main__":
    main()
