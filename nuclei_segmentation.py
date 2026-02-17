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

def ask_params_for_image(img_title):
    gd = GenericDialog("Nuclei segmentation params")
    gd.addMessage("Set parameters for nuclei segmentation.")

    gd.addNumericField("DAPI channel (1-based):", 1, 0)
    gd.addNumericField("Measurement channel (1-based):", 2, 0)

    gd.addChoice("Threshold method:", ["Triangle","Otsu","Huang","Yen","Li","Moments","Default"], "Otsu")

    gd.addNumericField("Min nucleus area (pixels^2):", 3000.0, 0)
    gd.addNumericField("Max nucleus area (pixels^2) (0 = no max):", 0.0, 0)

    gd.addNumericField("Min circularity (0..1):", 0.3, 2)
    gd.addNumericField("Max circularity (0..1):", 1.0, 2)

    gd.addNumericField("Gaussian Blur Sigma (1..5):", 1.5, 1)
    gd.addNumericField("Number of erosion steps (0...5):", 3, 0)
    gd.addNumericField("Number of dilation steps (0...5):", 5, 0)

    gd.addCheckbox("Apply background subtraction", True)
    gd.addNumericField("Background value (rolling ball radius or constant):", 10, 0)

    gd.addCheckbox("Exclude edge particles", True)
    gd.addCheckbox("Fill holes", True)
    gd.addCheckbox("Single ROI per image", True)

    gd.showDialog()
    if gd.wasCanceled():
        return None

    params = {}
    params["DAPI_CHANNEL"] = int(gd.getNextNumber())
    params["MEASURE_CHANNEL"] = int(gd.getNextNumber())
    params["thr_method"] = gd.getNextChoice()
    params["min_area"] = float(gd.getNextNumber())
    params["max_area"] = float(gd.getNextNumber())
    params["min_circularity"] = float(gd.getNextNumber())
    params["max_circularity"] = float(gd.getNextNumber())
    params["gaussian_blur_sigma"] = float(gd.getNextNumber()) 
    params["erosion_steps"] = int(gd.getNextNumber())
    params["dilation_steps"] = int(gd.getNextNumber())
    params["do_bg_subtraction"] = bool(gd.getNextBoolean())
    params["bg_value"] = float(gd.getNextNumber())
    params["exclude_edges"] = bool(gd.getNextBoolean())
    params["fill_holes"] = bool(gd.getNextBoolean())
    params["single_roi"] = bool(gd.getNextBoolean())

    return params

def get_active_image():
    """
    Returns the currently active ImagePlus in Fiji.
    Stops the script if no image is open.
    """
    imp = IJ.getImage()

    if imp is None:
        IJ.error("No active image found.")
        raise SystemExit

    return imp

def _img_name_processing(name):
    try:
        if "MP" in name and " - " in name:
            name = name.split("-")[1] # split string
            name = name.replace(" ", "", 1) # delete fist blank in the string
            name = name.replace(" ", "_") # repalce other blanks to underscore
        else:
            name = os.path.splitext(name)[0] # delete extention
        return name
    except Exception as e:
         raise Exception("ERROR in parsing image name")
    
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

    # Sort by channel number: C1, C2, C3...
    def chan_index(t):
        # expects "C2-..." -> 2
        try:
            return int(t.split("-")[0][1:])
        except:
            return 999
    split_imps.sort(key=lambda im: chan_index(im.getTitle()))

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
    
def _close_all_images_except(keep_imp):
    """
    Closes all image windows except keep_imp.
    'changes=False' prevents 'Save changes?' dialogs.
    """
    ids = WindowManager.getIDList()
    if not ids:
        return

    for wid in ids:
        imp = WindowManager.getImage(wid)
        if imp is None:
            continue
        if imp != keep_imp:
            imp.changes = False
            imp.close()

def close_images(imps):
    for im in imps:
        if im is None:
            continue
        im.changes = False
        im.close()
         
def close_results_table():
	"""
    Closes the standard ImageJ 'Results' table window if it exists.
    Uses dispose() because Results is usually a Swing window.
    """
	w = WindowManager.getWindow("Results")
	if w is None:
		return
	w.dispose()
    

def build_mask_from_rois(reference_imp, rm):
    """
    Creates an 8-bit mask image where pixels inside each ROI are filled with 255 (white),
    and background is 0 (black).
    This produces a 'particles-only' mask based on the ROIs that passed Analyze Particles.
    """
    w = reference_imp.getWidth()
    h = reference_imp.getHeight()

    mask = IJ.createImage("Nuclei_mask_particles_only", "8-bit black", w, h, 1)
    ip = mask.getProcessor()
    ip.setValue(255)	# fill value (white)
    
 	# Fill each ROI into the mask
    for i in range(rm.getCount()):
        roi = rm.getRoi(i)
        mask.setRoi(roi)
        ip.fill(mask.getRoi())

    mask.killRoi()
    return mask

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

def process_image(imp, p):
    '''
    This function process a single image
    imp - image
    p - parameters
    '''
    # Parameteres
    DAPI_CHANNEL = p["DAPI_CHANNEL"]
    MEASURE_CHANNEL = p["MEASURE_CHANNEL"]
    thr_method = p["thr_method"]
    min_area = p["min_area"]
    max_area = p["max_area"]
    min_circularity = p["min_circularity"]
    max_circularity = p["max_circularity"]
    gaussian_blur_sigma = p["gaussian_blur_sigma"]
    erosion_steps = p["erosion_steps"]
    dilation_steps = p["dilation_steps"]
    substruct_bg = p["do_bg_subtraction"] # bppl
    bg_radius = p["bg_value"]
    exclude_edges = p["exclude_edges"] # bool
    fill_holes = p["fill_holes"] # bool
    single_roi = p["single_roi"] # bool

    # Processing image title
    img_title = imp.getTitle()
    img_title = img_name_processing(img_title)

    # Initialize/reset ROI Manager so we start clean
    rm = ensure_roi_manager(reset=True)

    # Split channels into separate images (C1, C2, ...)
    split_imps = split_channels(imp)

    # Select DAPI channel image (used for nuclei segmentation)
    dapi_imp = pick_channel_by_index(split_imps, DAPI_CHANNEL)

    # Select the measurement channel image (used for mean intensity measurement)
    meas_imp = pick_channel_by_index(split_imps, MEASURE_CHANNEL)

    if dapi_imp is None or meas_imp is None:
        IJ.error("Missing channels for: " + img_title)
        close_images(split_imps)
        return
    
    # --- Background substurction in MEASUREMENT channel ---
    if substruct_bg:
        subtract_background(meas_imp, bg_radius, light_background=False, use_paraboloid=False, do_presmooth=True)

    # --- NUCLEI SEGMENTATION ON DAPI

    # Work on a duplicate so we don’t modify the original DAPI channel image
    dapi_work = dapi_imp.duplicate()
    dapi_work.setTitle("DAPI_work")
    dapi_work.show()

    # Preprocessing: helps reduce uneven background and noise
    IJ.run(dapi_work, "Gaussian Blur...", "sigma={}".format(gaussian_blur_sigma))

    # Thresholding: create a binary mask from the DAPI channel
    # "{} dark" assumes nuclei are bright on a dark background
    IJ.setAutoThreshold(dapi_work, "{} dark".format(thr_method))
    IJ.run(dapi_work, "Convert to Mask", "")

    # Post-processing: fill holes inside nuclei
    if fill_holes:
        IJ.run(dapi_work, "Fill Holes", "")
    
    # Make erosion to separate close nuclei (optional, can be adjusted by user)
    if erosion_steps > 0:
        for i in range(erosion_steps):
            IJ.run("Erode")
    # Make dilation to restore original size after erosion (optional, can be adjusted by user)
    if dilation_steps > 0:
        for i in range(n):
            IJ.run("Dilate")

    # --- ANALYZE PARTICLES -> ROIs IN ROI MANAGER

    # ParticleAnalyzer:
    # - ADD_TO_MANAGER adds each detected particle as an ROI
    # - measurements only used during detection (we’ll measure on C2 later)
    options = ParticleAnalyzer.ADD_TO_MANAGER
    if exclude_edges:
        options |= ParticleAnalyzer.EXCLUDE_EDGE_PARTICLES
        
    measurements = Measurements.AREA

    # If your Fiji throws an error here, switch to IJ.run("Analyze Particles...", ...) instead.
    pa = ParticleAnalyzer(options, measurements, None,
                        float(min_area),
                        (float(max_area) if max_area and max_area > 0 else float("inf")),
                        min_circularity, max_circularity)

    ok = pa.analyze(dapi_work)
    if (not ok) or rm.getCount() == 0:
        IJ.log("No nuclei found for: " + img_title)
        dapi_work.changes = False
        dapi_work.close()
        close_images(split_imps)
        return
    
    if single_roi:
        max_area = -1.0
        max_roi = None

        for i in range(rm.getCount()):
                roi = rm.getRoi(i)
                if roi is None:
                    continue
                imp.setRoi(roi)
                stats = imp.getStatistics(Measurements.AREA)
                area = stats.area
                if area > max_area:
                    max_area = area
                    max_roi = roi
        if max_roi is None:
             raise Exception("Could not find a valid ROI")
        # Keep only the biggest ROI in ROI Manager
        rm.reset()
        rm.addRoi(max_roi)
    
    # --- Save measurement channel image ---
    MEASURE_CHANNEL_name = "C{}_{}.jpeg".format(MEASURE_CHANNEL, img_title)
    MEASURE_CHANNEL_path = os.path.join(output_dir, MEASURE_CHANNEL_name)
    meas_imp.show()
    IJ.save(meas_imp, MEASURE_CHANNEL_path)

    # Save ROIs as a separate file .zip
    roi_path = os.path.join(output_dir, "C{}_{}_rois.zip".format(DAPI_CHANNEL, img_title))
    rm.runCommand("Save", roi_path)
    
    # --- SAVE MASK OF ACCEPTED NUCLEI (ROIs-ONLY MASK)
    mask_particles = build_mask_from_rois(dapi_work, rm)
    mask_particles.show()
    mask_particles.updateAndDraw()

    mask_path = os.path.join(output_dir, "C{}_{}_mask.jpeg".format(DAPI_CHANNEL, img_title))
    IJ.save(mask_particles, mask_path)

    # --- Measure on measurement channel ---
    IJ.run("Set Measurements...", "area mean decimal=3")  # no redirect
    IJ.run("Clear Results", "")
    rm.runCommand(meas_imp, "Measure")

    # Save Results as CSV
    results_path = os.path.join(output_dir, "C{}_{}_roi.csv".format(MEASURE_CHANNEL, img_title))
    IJ.saveAs("Results", results_path)
    close_results_table()

    # --- Cleanup ONLY what we created ---
    dapi_work.changes = False
    dapi_work.close()

    mask_particles.changes = False
    mask_particles.close()

    close_images(split_imps)  # closes C1-..., C2-..., etc. for THIS image only

    # Reset and close ROI manager
    cleanup_iteration()

    IJ.log("Done: " + imp.getTitle())

# --- Main ---

# Check if at least one image is opened
ids = WindowManager.getIDList()
if not ids:
    IJ.error("No images open.")
    raise SystemExit

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
    raise SystemExit

# Keep only unique images
unique_images = list(set(images))
n = len(unique_images) # total amount of images to process

# Ask user where to save outputs
output_dir = IJ.getDirectory("Choose a directory to save data")
if output_dir is None:
    IJ.error("No output directory selected!")
    raise SystemExit

errors = []  # collect all errors here

# Ask user about the parameters
params = ask_params_for_image(imp.getTitle())
if params is None:
    IJ.error("No parameters provided!")
    raise SystemExit
    
# ---- Loop: show GUI per image, then process ----
for call_id, imp in enumerate(unique_images, start=1):
    # Make Log message
    msg = "Processing {}/{}: {}".format(call_id, n, imp.getTitle())
    IJ.log(msg)

    try:
        process_image(imp, params)

    except Exception as e:
        # log immediately
        IJ.log("ERROR in {}: {}".format(imp.getTitle(), e))
        IJ.log(traceback.format_exc())  # comment out if too verbose
        continue

    finally:
        # clean-up ROI manager
        cleanup_iteration()

# ---- After the loop: print a summary ----
IJ.log("===== RUN SUMMARY: {} error(s) =====".format(len(errors)))
for k, er in enumerate(errors, start=1):
    IJ.log("#{k} [{id}] {title} | {type}: {msg}".format(
        k=k, id=er["id"], title=er["title"], type=er["type"], msg=er["msg"]
    ))

# Finish progress
IJ.log("Analysis is finished!")