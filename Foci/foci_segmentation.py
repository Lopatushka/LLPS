from ij import IJ
from ij.plugin.frame import RoiManager
from ij import ImagePlus
from ij.gui import GenericDialog
from ij import WindowManager
import os
import re

def check_dir(dir):
	if dir is None:
		IJ.error("No directory selected!")
		raise SystemExit
        
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
    gd.addStringField("Threshold expression:", "std(Wave.F1)", 20)

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

    # ---- Camera parameters ----
    gd.addNumericField("Pixel size:", 58.7, 1)
    gd.addNumericField("Photoelectrons per ADU:", 1.0, 1)
    gd.addNumericField("Quantum efficiency (0..1):", 1, 1)
    gd.addNumericField("ADU offset:", 0, 1)
    gd.addNumericField("Electrons/pixel:", 1, 1)
    gd.addNumericField("EMCCD gain:", 1, 1)

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

    p["pixel_size"] = float(gd.getNextNumber())
    p["photoelectrons_per_adu"] = float(gd.getNextNumber())
    p["quantum_efficiency"] = float(gd.getNextNumber())
    p["base_level"] = float(gd.getNextNumber())
    p["readout_noise"] = float(gd.getNextNumber())
    p["em_gain"] = float(gd.getNextNumber())

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

def safe_name(s):
    """Make a string safe for filenames."""
    s = str(s)
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = s.replace(" ", "_")
    return s

def close_window(title):
    win = WindowManager.getWindow(title)
    if win: win.dispose()

def close_all_images():
    ids = WindowManager.getIDList()
    if ids is not None:
        for image_id in ids:
            imp = WindowManager.getImage(image_id)

            if imp is not None:
                imp.changes = False   # avoid "Save changes?" dialog
                imp.close()

def foci_image(imp, parameters, output_dir):
    """
    Process a single image.

    imp  : ImagePlus
    p    : dict-like parameters (optional, used later)
    """
    img_name = imp.getTitle()
    img_base = safe_name(os.path.splitext(img_name)[0])
    
    # Duplicate image
    dup = imp.duplicate()
    dup.setTitle("{}_foci".format(img_base))
    dup_title = dup.getTitle()
    dup.show()

    # Convert image into 16-bit if needed
    dup_type = dup.getType()
    if dup_type not in (ImagePlus.GRAY8, ImagePlus.GRAY16):
        IJ.run(dup, "16-bit", "")
        dup.changes = False
    
    # Run ThunderSTORM for the image
    IJ.run(dup, "Run analysis", parameters)

    # Check that ThunderSTORM output exists
    if WindowManager.getWindow("ThunderSTORM: results") is None:
        raise RuntimeError("ThunderSTORM results window not found (analysis may have failed).")

    # --- Save results ---
    # Export CSV
    csv_path = os.path.abspath(os.path.join(output_dir, "{}_foci.csv".format(img_base)))
    csv_path_ij = csv_path.replace("\\", "/")
    export_opts = (
        'filepath=[{}] '
        'fileformat=[CSV (comma separated)] '
        'sigma=true intensity=true chi2=false offset=false saveprotocol=false '
        'x=true y=true bkgstd=false id=true uncertainty=false frame=false'
    ).format(csv_path_ij)

    IJ.selectWindow("ThunderSTORM: results")
    IJ.run("Export results", export_opts)

    # Save foci image
    foci_path = os.path.join(output_dir, "{}.png".format(dup_title))
    IJ.save(dup, foci_path)

    # -- Closed windows ---
    close_window("ThunderSTORM: results")
    close_all_images()



def _foo():
    for i, roi in enumerate(rois):
        dup = None
        roi_name = None
        try:
            roi_name = roi.getName()
            if roi_name is None:
                roi_name = "roi_{:02d}".format(i + 1)
            roi_base = safe_name(roi_name)

            IJ.log("Processing image: {} and ROI: {}".format(img_name, roi_name))

            # Make sure old results window doesn't interfere
            close_window("ThunderSTORM: results")

            # Set ROI and clear data outside ROI
            dup = imp.duplicate()
            dup.show()
            dup.setRoi(roi)
            dup.setTitle("ROI_{:02d}_{}".format(i + 1, img_name))
            IJ.run(dup, "Clear Outside", "")
            dup.killRoi()          

            # Convert to 16-bit only if needed. Optional
            dup_type = dup.getType()
            if dup_type not in (ImagePlus.GRAY8, ImagePlus.GRAY16):
                IJ.run(dup, "16-bit", "")
                dup.changes = False

            IJ.run(dup, "Run analysis", parameters)

            # ---- Export CSV ----
            csv_path = os.path.abspath(os.path.join(output_dir, "{}_{}.csv".format(img_base, roi_base)))
            csv_path_ij = csv_path.replace("\\", "/")

            export_opts = (
                'filepath=[{}] '
                'fileformat=[CSV (comma separated)] '
                'sigma=true intensity=true chi2=false offset=false saveprotocol=false '
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
def main():
    # Ask user about the directory with data to process
    input_dir = IJ.getDirectory("Choose a directory with data to process")
    check_dir(input_dir)

    # Open images from the Input directory
    exts = (".tif", ".tiff")

    # List of images with desired extension and filtration
    images = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and f.lower().endswith(exts)
        and "ROI" in f
    ]
    images.sort()

    # Number of founded images to process
    n = len(images)

    IJ.log("Found {} images.".format(n))
    
    if n == 0:
        IJ.error("No images found in the directory! Please check the directory. Exiting.")
        return

    # Set ThunderSTORM parameters once for all images
    ts_params = ask_params_for_thunderstorm()
    if ts_params is None:
        IJ.log("Parameters for ThunderSTORM are not set. Exiting.")
        return
    ts_opts = thunderstorm_options(ts_params)

    # Ask user where to save outputs
    output_dir = IJ.getDirectory("Choose a directory to save data")
    check_dir(output_dir)

    # --- Iterate over images ---
    for call_id, filename in enumerate(images, start=1):
        # Open image
        path = os.path.join(input_dir, filename)
        imp = IJ.openImage(path)

        # Check that there is image
        if imp is None:
            IJ.log("SKIP (cannot open image): " + title)
            continue

        title = imp.getTitle()

        # Make Log message
        msg = "Processing {}/{}: {}".format(call_id, n, title)
        IJ.log(msg)

        # Show image
        imp.show()

        try:
            foci_image(imp, ts_opts, output_dir)

        except Exception as e:
            IJ.log("IMAGE ANALYSIS FAILED {}: {}".format(title, e))

    IJ.log("Analysis is finished!")

main()