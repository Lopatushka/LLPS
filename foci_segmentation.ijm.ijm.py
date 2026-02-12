from ij import IJ
from ij.plugin.frame import RoiManager
from ij import ImagePlus
from ij.gui import GenericDialog
from ij import WindowManager
import os
import re
import csv

def check_dir(dir):
	if dir is None:
		IJ.error("No directory selected!")
		raise SystemExit
	
def make_key(filename):
    s = filename.lower()

    # remove leading channel prefix like "c1_" or "c2_"
    s = re.sub(r'^c\d+_', '', s)

    # remove roi suffix
    s = re.sub(r'_rois(?=\.|$)', '', s)

    # strip known extensions repeatedly (handles ".nd2.jpg", ".ome.tif", etc.)
    while True:
        new = re.sub(r'(\.ome)?\.(tif|tiff|png|jpg|jpeg|zip|nd2|czi|lif)$', '', s)
        if new == s:
            break
        s = new

    # cleanup leftover underscores/spaces
    s = re.sub(r'[\s_]+$', '', s)
    s = re.sub(r'^[\s_]+', '', s)

    return s

def img_roi_pairs(images, rois):
	roi_map = {}
	for r in rois:
		k = make_key(r)
		roi_map.setdefault(k, []).append(r)

	pairs = []
	unmatched_images = []

	for img in images:
		k = make_key(img)
		if k in roi_map and len(roi_map[k]) > 0:
			roi_file = roi_map[k].pop(0)  # take one matching roi
			pairs.append((img, roi_file))
		else:
			unmatched_images.append(img)

	return pairs, unmatched_images

def cleanup_iteration():
    rm = RoiManager.getInstance()
    if rm is not None:
        rm.reset()
        rm.close()
        
def ask_params_for_thunderstorm():
    """
    Ask ThunderSTORM parameters ONCE (to reuse for all images).
    Returns: dict or None if canceled.
    """
    gd = GenericDialog("ThunderSTORM parameters (apply to ALL images)")

    # ---- Filter ----
    gd.addChoice(
        "Filter:",
        ["Wavelet filter (B-Spline)", "Gaussian filter"],
        "Wavelet filter (B-Spline)"
    )
    gd.addNumericField("Wavelet scale:", 2.0, 1)
    gd.addNumericField("Wavelet order:", 3, 0)

    # ---- Detector ----
    gd.addChoice("Detector:", ["Local maximum"], "Local maximum")
    gd.addChoice("Connectivity:", ["4-neighbourhood", "8-neighbourhood"], "8-neighbourhood")
    gd.addStringField("Threshold expression:", "2*std(Wave.F1)", 20)

    # ---- Estimator ----
    gd.addChoice("Estimator:", ["PSF: Integrated Gaussian"], "PSF: Integrated Gaussian")
    gd.addNumericField("PSF sigma:", 1.6, 2)
    gd.addNumericField("Fit radius (pixels):", 3, 0)
    gd.addChoice("Fitting method:", ["Weighted Least squares", "Least squares"], "Weighted Least squares")

    # ---- Options ----
    gd.addCheckbox("Full image fitting", False)
    gd.addCheckbox("Enable MFA", False)

    # ---- Renderer ----
    gd.addChoice("Renderer:", ["No Renderer", "Gaussian rendering"], "No Renderer")

    gd.showDialog()
    if gd.wasCanceled():
        return None

    # IMPORTANT: read values in the same order as added
    p = {}
    p["filter"] = gd.getNextChoice()
    p["scale"] = float(gd.getNextNumber())
    p["order"] = int(gd.getNextNumber())

    p["detector"] = gd.getNextChoice()
    p["connectivity"] = gd.getNextChoice()
    p["threshold"] = gd.getNextString()

    p["estimator"] = gd.getNextChoice()
    p["sigma"] = float(gd.getNextNumber())
    p["fitradius"] = int(gd.getNextNumber())
    p["method"] = gd.getNextChoice()

    p["full_image_fitting"] = bool(gd.getNextBoolean())
    p["mfaenabled"] = bool(gd.getNextBoolean())

    p["renderer"] = gd.getNextChoice()

    return p

def thunderstorm_options(p):
    """
    Build a ThunderSTORM 'Run analysis' macro options string from parameters dict p.
    Output is safe (spaces between options; dropdown values in brackets; booleans lower-case).
    """
    def b(x):
        return "true" if bool(x) else "false"

    opts = [
        'filter=[{}]'.format(p["filter"]),
        'scale={}'.format(p["scale"]),
        'order={}'.format(p["order"]),
        'detector=[{}]'.format(p["detector"]),
        'connectivity={}'.format(p["connectivity"]),
        'threshold={}'.format(p["threshold"]),
        'estimator=[{}]'.format(p["estimator"]),
        'sigma={}'.format(p["sigma"]),
        'fitradius={}'.format(p["fitradius"]),
        'method=[{}]'.format(p["method"]),
        'full_image_fitting={}'.format(b(p["full_image_fitting"])),
        'mfaenabled={}'.format(b(p["mfaenabled"])),
        'renderer=[{}]'.format(p["renderer"]),
    ]
    return " ".join(opts)

def _safe_name(s):
    """Make a string safe for filenames."""
    s = str(s)
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = s.replace(" ", "_")
    return s

def close_window(title):
    win = WindowManager.getWindow(title)
    if win: win.dispose()

def foci_image(imp, rois, parameters, output_dir):
    """
    Process a single image for multiple ROIs.

    imp  : ImagePlus
    rois : list of Roi objects
    p    : dict-like parameters (optional, used later)
    """
    img_name = imp.getTitle()
    img_base = _safe_name(os.path.splitext(img_name)[0])

    for i, roi in enumerate(rois):
        dup = None
        roi_name = None
        try:
            roi_name = roi.getName()
            if roi_name is None:
                roi_name = "roi_{:02d}".format(i + 1)
            roi_base = _safe_name(roi_name)

            IJ.log("Processing image: {} and ROI: {}".format(img_name, roi_name))

            # Make sure old results window doesn't interfere
            close_window("ThunderSTORM: results")

            # Set ROI and crop
            #imp.setRoi(roi)
            dup = imp.duplicate()
            #dup = imp.crop()
            dup.show()
            dup.setRoi(roi)
            dup.setTitle("ROI_{:02d}_{}".format(i + 1, img_name))
            IJ.run(dup, "Clear Outside", "")
            dup.killRoi()          

            # Convert to 16-bit only if needed
            dup_type = dup.getType()
            if dup_type not in (ImagePlus.GRAY8, ImagePlus.GRAY16):
                IJ.run(dup, "16-bit", "")
                dup.changes = False

            # ---- Run ThunderSTORM ----
            IJ.run(dup, "Run analysis", parameters)

            # ---- Export CSV ----
            csv_path = os.path.abspath(os.path.join(output_dir, "{}_{}.csv".format(img_base, roi_base)))
            csv_path_ij = csv_path.replace("\\", "/")

            export_opts = (
                'filepath=[{}] '
                'fileformat=[CSV (comma separated)] '
                'sigma=false intensity=true chi2=false offset=false saveprotocol=false '
                'x=true y=true bkgstd=false id=true uncertainty=false frame=false'
            ).format(csv_path_ij)
            
            # Select results and export
            if WindowManager.getWindow("ThunderSTORM: results") is None:
                raise RuntimeError("ThunderSTORM results window not found (analysis may have failed).")
            
            IJ.selectWindow("ThunderSTORM: results")
            IJ.run("Export results", export_opts)

            # Save cropped image
            cropped_path = os.path.join(output_dir, "{}_{}.png".format(img_base, roi_name))
            IJ.save(dup, cropped_path)

        except Exception as e:
            IJ.log(
                "Error on ROI {} ({}): {}".format(
                    i + 1,
                    roi_name if roi_name is not None else "?",
                    e
                )
            )

        finally:
            close_window("ThunderSTORM: results")
            if dup is not None:
                dup.close()
            #imp.killRoi()    

# --- Main ---
# Ask user about the directory with data to process
input_dir = IJ.getDirectory("Choose a directory with data to process")
check_dir(input_dir)

# Open images from the Input directory
exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# List of images with desired extension and filtration
images = [
    f for f in os.listdir(input_dir)
    if os.path.isfile(os.path.join(input_dir, f))
    and f.lower().endswith(exts)
    and "mask" not in f.lower()
]
images.sort()
n_images = len(images)

# List of ROIs for the corresponding images
rois = [
    f for f in os.listdir(input_dir)
    if os.path.isfile(os.path.join(input_dir, f))
    and f.lower().endswith('.zip')
	and 'rois' in f.lower()
]
rois.sort()
n_rois = len(rois)

if n_images == 0:
	IJ.error("No images found in the directory! Please check the directory.")
	raise SystemExit

# Match images and ROIs based on filename keys
pairs, unmatched_images = img_roi_pairs(images, rois)
if len(pairs) == 0:
	IJ.error("No matching image-ROI pairs found! Please check the directory.")
	raise SystemExit
if unmatched_images:
    IJ.log("Images without ROI: {}".format(len(unmatched_images)))

IJ.log("Found {} image-ROI pairs to process.".format(len(pairs)))

# Set ThunderSTORM parameters once for all images
ts_params = ask_params_for_thunderstorm()
if ts_params is None:
    IJ.log("Parameters for ThunderSTORM are not set. Exiting.")
    raise SystemExit
ts_opts = thunderstorm_options(ts_params)

# Ask user where to save outputs
output_dir = IJ.getDirectory("Choose a directory to save data")
check_dir(output_dir)

# ---- open subsequently in Fiji ----
rm = RoiManager.getRoiManager()

# Iterate over matched pairs of ROI and images
for img, roi in pairs:
    IJ.log("Open image: " + img)

    # Open the image
    imp = IJ.openImage(os.path.join(input_dir, img))
    if imp is None:
        IJ.log("SKIP (cannot open image): " + img)
        continue

    imp.show()

    # Reset ROI Manager before loading new ROIs
    rm.reset()
    IJ.log("Open ROI: " + roi)

    # Open the ROI zip file (loads all ROIs into the manager)
    roi_path = os.path.join(input_dir, roi)
    rm.open(roi_path)

    try:
        # Get ROIs AFTER loading them
        rois = list(rm.getRoisAsArray())
        foci_image(imp, rois, ts_opts, output_dir)

    except Exception as e:
        IJ.log("IMAGE FAILED {}: {}".format(img, e))

    finally:
        if imp is not None:
            imp.close()
        rm.reset()

# Fininsh up and close everything
cleanup_iteration()

IJ.log("Analysis is finished!")



	