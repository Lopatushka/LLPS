# ============================================================
# PACKAGES IMPORT
# ============================================================
from ij import IJ, WindowManager
from ij.gui import GenericDialog
from ij.plugin.frame import RoiManager
from ij.measure import Measurements, ResultsTable
from ij.plugin.filter import ParticleAnalyzer
from ij.process import AutoThresholder
import os
import csv

# ============================================================
# PARAMETERS (edit these)
# ============================================================

#DAPI_CHANNEL = 1	     # 1-based channel index for DAPI (after Split Channels): C1 = 1, C2 = 2, ...          
#MEASURE_CHANNEL = 2	     # 1-based channel index to measure mean intensity from (e.g., C2)       
#thr_method = "Triangle"	 # Auto-threshold method for DAPI segmentation (e.g., Otsu, Triangle, Huang, ...)
#min_area = 200	         # Minimum nucleus area in pixels^2
#max_area = None	         # Maximum nucleus area in pixels^2 (None/0 => no max)            
#min_circularity = 0.2	 # Min circularity filter for nuclei (0..1)
#max_circularity = 1.0	 # Max circularity filter for nuclei (0..1)
#exclude_edges = True	 # True = exclude edge particles, False = keep them
#output_dir = "D:/test"	 # Directory to save results

# ============================================================
# FUNCTIONS
# ============================================================

def is_original_image(imp):
    title = imp.getTitle()
    return not (
        title.startswith("C") and "-" in title or
        title.endswith("_mask.tif") or
        title == "DAPI_work"
    )

def ask_params_for_image(img_title):
    gd = GenericDialog("Nuclei segmentation params")
    gd.addMessage("Image: " + img_title)

    gd.addNumericField("DAPI channel (1-based):", 1, 0)
    gd.addNumericField("Measurement channel (1-based):", 2, 0)
    gd.addChoice("Threshold method:", ["Triangle","Otsu","Huang","Yen","Li","Moments","Default"], "Triangle")
    gd.addNumericField("Min nucleus area (pixels^2):", 200.0, 0)
    gd.addNumericField("Max nucleus area (pixels^2) (0 = no max):", 0.0, 0)
    gd.addNumericField("Min circularity (0..1):", 0.2, 2)
    gd.addNumericField("Max circularity (0..1):", 1.0, 2)
    gd.addCheckbox("Exclude edge particles", True)

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
    params["exclude_edges"] = bool(gd.getNextBoolean())

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
    
def base_name(title):
    """
    Removes the file extension from the image title safely.
    Example: 'cell1.tif' -> 'cell1'
    """
    return os.path.splitext(title)[0]
    
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
    
    # Split channels: creates new windows like "C1-<orig_title>", "C2-<orig_title>", ...
    IJ.run(imp, "Split Channels", "")
    
    # Get all currently opened image window IDs
    ids = WindowManager.getIDList()
    if not ids:
        IJ.error("No windows after Split Channels.")
        raise SystemExit

    split_imps = []
    for wid in ids:
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
    
def close_all_images_except(keep_imp):
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

def process_image(imp, p):
    '''
    This function process a single image
    '''
    # Initialize parameteres
    DAPI_CHANNEL = p["DAPI_CHANNEL"]
    MEASURE_CHANNEL = p["MEASURE_CHANNEL"]
    thr_method = p["thr_method"]
    min_area = p["min_area"]
    max_area = p["max_area"]
    min_circularity = p["min_circularity"]
    max_circularity = p["max_circularity"]
    exclude_edges = p["exclude_edges"]

    # Get the currently active image (multichannel)
    imp = get_active_image()
    img_title = imp.getTitle()
    img_base = base_name(img_title)

    # Initialize/reset ROI Manager so we start clean
    rm = ensure_roi_manager(reset=True)

    # Split channels into separate images (C1, C2, ...)
    split_imps = split_channels(imp)

    # Select DAPI channel image (used for nuclei segmentation)
    dapi_imp = pick_channel_by_index(split_imps, DAPI_CHANNEL)

    # Select the measurement channel image (used for mean intensity measurement)
    meas_imp = pick_channel_by_index(split_imps, MEASURE_CHANNEL)

    if dapi_imp is None:
        IJ.error("Cannot pick DAPI channel {}. Check number of channels.".format(DAPI_CHANNEL))
        raise SystemExit
    if meas_imp is None:
        IJ.error("Cannot pick measurement channel {}. Check number of channels.".format(MEASURE_CHANNEL))
        raise SystemExit
    
    # Save MEASURE_CHANNEL image
    meas_imp.changes = False
    MEASURE_CHANNEL_name = "{}_measure_channel.jpeg".format(img_base)
    MEASURE_CHANNEL_path = os.path.join(output_dir, MEASURE_CHANNEL_name)
    meas_imp.show()
    IJ.save(meas_imp, MEASURE_CHANNEL_path)

    # ------------------------------------------------------------
    # 1) NUCLEI SEGMENTATION ON DAPI
    # ------------------------------------------------------------

    # Work on a duplicate so we don’t modify the original DAPI channel image
    dapi_work = dapi_imp.duplicate()
    dapi_work.setTitle("DAPI_work")
    dapi_work.show()

    # Preprocessing: helps reduce uneven background and noise
    IJ.run(dapi_work, "Subtract Background...", "rolling=50")
    IJ.run(dapi_work, "Gaussian Blur...", "sigma=1")

    # Thresholding: create a binary mask from the DAPI channel
    # "{} dark" assumes nuclei are bright on a dark background
    IJ.setAutoThreshold(dapi_work, "{} dark".format(thr_method))
    IJ.run(dapi_work, "Convert to Mask", "")

    # Post-processing: fill holes inside nuclei
    IJ.run(dapi_work, "Fill Holes", "")

    # ------------------------------------------------------------
    # 2) ANALYZE PARTICLES -> ROIs IN ROI MANAGER
    # ------------------------------------------------------------

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
    if not ok:
        IJ.error("ParticleAnalyzer returned False (no particles or error).")
        raise SystemExit

    if rm.getCount() == 0:
        IJ.error("No nuclei found (ROI Manager empty). Check threshold/morphology/min_area.")
        raise SystemExit

    # ------------------------------------------------------------
    # 3) SAVE MASK OF ACCEPTED NUCLEI (ROIs-ONLY MASK)
    # ------------------------------------------------------------

    mask_particles = build_mask_from_rois(dapi_work, rm)
    mask_particles.show()
    mask_particles.updateAndDraw()

    mask_path = os.path.join(output_dir, "{}_nuclei_mask.tif".format(img_base))
    IJ.save(mask_particles, mask_path)

    # ------------------------------------------------------------
    # 4) MEASURE AREA + MEAN INTENSITY ON CHANNEL 2 FOR THESE ROIs
    # ------------------------------------------------------------

    # Close everything except the measurement channel to reduce confusion/state
    # close_all_images_except(meas_imp)

    # Ensure measurement channel is active (important for consistent measurements)
    imp = get_active_image()
    img_title = imp.getTitle()
    img_base = base_name(img_title)

    print("Processing the image: {}".format(img_title))

    # Set measurements for the Results table:
    # - area + mean intensity
    # - NO redirect (redirect settings can break on repeated runs)
    IJ.run("Set Measurements...", "area mean redirect=None decimal=3")

    # Clear old results so you only keep this run
    IJ.run("Clear Results", "")

    # Measure all ROIs on meas_imp -> fills the standard Results table
    rm.runCommand(meas_imp, "Measure")

    # Save Results as CSV
    results_path = os.path.join(output_dir, "{}_C2_roi_area_mean.csv".format(img_base))
    IJ.saveAs("Results", results_path)

    # Close Results window (optional)
    close_results_table()

    # Close MEASURED image
    meas_imp.changes = False
    meas_imp.close()

    IJ.log("Done: " + imp.getTitle())

# ============================================================
# MAIN
# ============================================================

# Check if at least one image is opened
ids = WindowManager.getIDList()
if not ids:
    IJ.error("No images open.")
    raise SystemExit

images = []
for wid in ids:
    imp = WindowManager.getImage(wid)
    if imp is None:
        continue
    title = imp.getTitle()
    # Skip typical derived images (adjust if needed)
    if (title.startswith("C") and "-" in title) or title in ["DAPI_work", "Nuclei_mask_particles_only"]:
        continue
    images.append(imp)

if not images:
    IJ.error("No suitable images found (only derived windows are open).")
    raise SystemExit

# Ask user where to save outputs
output_dir = IJ.getDirectory("Choose a directory to save data")
if output_dir is None:
    IJ.error("No output directory selected.")
    raise SystemExit

# ---- loop: show GUI per image, then process ----
for imp in images:
    if is_original_image(imp):

        params = ask_params_for_image(imp.getTitle())

        if params is None:
            IJ.log("Canceled on image: " + imp.getTitle())
            break

        IJ.log("Processing: " + imp.getTitle())
        process_image(imp, params)