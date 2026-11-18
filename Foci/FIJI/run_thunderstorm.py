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
    # Only the wavelet filter is offered: scale/order below and the default
    # threshold expression std(Wave.F1) are specific to it.
    gd.addChoice(
        "Filter:",
        ["Wavelet filter (B-Spline)"],
        "Wavelet filter (B-Spline)"
    )
    gd.addNumericField("Wavelet scale:", 2.0, 1)
    gd.addNumericField("Wavelet order:", 3, 0)

    # ---- Detector ----
    gd.addChoice("Detector:", ["Local maximum"], "Local maximum")
    gd.addChoice("Connectivity:", ["4-neighbourhood", "8-neighbourhood"], "8-neighbourhood")
    gd.addStringField("Threshold expression:", "std(Wave.F1)", 10)

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

    # ---- Camera parameters (applied via ThunderSTORM "Camera setup") ----
    gd.addMessage("Pixel size is taken from each image's calibration (nm/px);\n"
                  "the value below is used only for uncalibrated images.")
    gd.addNumericField("Fallback pixel size (nm):", 57.5, 2)
    gd.addNumericField("Photoelectrons per ADU:", 1.0, 2)
    gd.addNumericField("Quantum efficiency (0..1):", 1, 2)
    gd.addNumericField("Base level / offset (ADU):", 0, 1)
    gd.addNumericField("Readout noise (e-/px):", 0, 2)
    gd.addNumericField("EM gain (1 = no EM gain):", 1, 1)

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
        'threshold=[{}]'.format(p["threshold"]),  # brackets: expression may contain spaces
        'estimator=[{}]'.format(p["estimator"]),
        'sigma={}'.format(p["sigma"]),
        'fitradius={}'.format(p["fitradius"]),
        'method=[{}]'.format(p["method"]),
        'full_image_fitting={}'.format(b(p["full_image_fitting"])),
        'mfaenabled={}'.format(b(p["mfaenabled"])),
        'renderer=[{}]'.format(p["renderer"]),
    ]
    return " ".join(opts)

def image_pixel_size_nm(imp, fallback_nm):
    """
    Pixel size in nm from the image calibration, or fallback_nm if the image
    is uncalibrated. Returns (pixel_size_nm, source_description).
    """
    cal = imp.getCalibration()
    unit = (cal.getUnit() or "").strip().lower()
    width = cal.pixelWidth
    if cal.scaled() and width > 0:
        if unit in ("micron", "microns", "um", u"µm", u"μm"):
            return width * 1000.0, "image calibration"
        if unit in ("nm", "nanometer", "nanometers"):
            return width, "image calibration"
    return fallback_nm, "fallback (image not calibrated in um/nm)"

def camera_setup(p, pixel_size_nm):
    """
    Apply camera parameters in ThunderSTORM. Must run before 'Run analysis',
    otherwise ThunderSTORM silently uses the last saved camera setup and all
    nm coordinates / sigma values are scaled by the wrong pixel size.
    """
    em = p["em_gain"] > 1
    opts = (
        "readoutnoise={} offset={} quantumefficiency={} isemgain={} "
        "photons2adu={} gainem={} pixelsize={}"
    ).format(
        p["readout_noise"], p["base_level"], p["quantum_efficiency"],
        "true" if em else "false",
        p["photoelectrons_per_adu"], p["em_gain"] if em else 1.0, pixel_size_nm
    )
    IJ.run("Camera setup", opts)

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

def foci_image(imp, parameters, camera, output_dir):
    """
    Process a single image.

    imp        : ImagePlus
    parameters : ThunderSTORM 'Run analysis' options string
    camera     : parameter dict from ask_params_for_thunderstorm (camera fields)
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
    
    # Camera setup with this image's pixel size, then run ThunderSTORM
    px_nm, px_source = image_pixel_size_nm(imp, camera["pixel_size"])
    IJ.log("  Pixel size: {:.3f} nm ({})".format(px_nm, px_source))
    camera_setup(camera, px_nm)
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
        # saveprotocol=true writes <name>_foci-protocol.txt with the camera
        # settings used; FOCI_STORM/map.py reads the pixel size from it.
        'sigma=true intensity=true chi2=false offset=false saveprotocol=true '
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

  
#--------------------------
# --------- MAIN ---------
#--------------------------
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
        and "roi" in f
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
            IJ.log("SKIP (cannot open image): " + filename)
            continue

        title = imp.getTitle()

        # Make Log message
        msg = "Processing {}/{}: {}".format(call_id, n, title)
        IJ.log(msg)

        # Show image
        imp.show()

        try:
            foci_image(imp, ts_opts, ts_params, output_dir)

        except Exception as e:
            IJ.log("IMAGE ANALYSIS FAILED {}: {}".format(title, e))

    IJ.log("Analysis is finished!")

main()