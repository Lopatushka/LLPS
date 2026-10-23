from ij import IJ, WindowManager
from ij.gui import GenericDialog, NonBlockingGenericDialog, WaitForUserDialog
from ij.plugin.frame import RoiManager
from ij.plugin import ChannelSplitter
import os
import re
import traceback
from ij.io import RoiEncoder
from java.io import FileOutputStream, ByteArrayOutputStream
from java.util.zip import ZipOutputStream, ZipEntry

def ask_params_for_image():
    gd = GenericDialog("Manual nuclei segmentation params")
    gd.addMessage("Set parameters for nuclei segmentation.")

    # Fields
    gd.addNumericField("DAPI channel (1-based):", 2, 0)
    gd.addNumericField("Measurement channel (1-based):", 1, 0)
    gd.addCheckbox('Exactly one ROI per image', False)
    
    gd.showDialog()
    if gd.wasCanceled():
        return None
    
    dapi = int(gd.getNextNumber())
    measure = int(gd.getNextNumber())
    one_roi = gd.getNextBoolean()
    
    if dapi < 1 or measure < 1:
        IJ.error('Channel numbers must be at least 1.')
        return None
    
    return {'DAPI_CHANNEL': dapi, 'MEASURE_CHANNEL': measure, 'one_roi': one_roi}


def _img_name_processing(name):
    try:
        if "MP" in name and " - " in name:
            if "Deconvolved" in name:
                name = name.split("-")[0] + "_" + name.split("-")[2]
                name = name.replace(" ", "", 1).replace(",", "").replace(" ", "_")
            else:
                name = name.split("-")[1] # split string
                name = name.replace(" ", "", 1) # delete fist blank in the string
                name = name.replace(" ", "_") # repalce other blanks to underscore
        name = os.path.splitext(name)[0] # delete extention
        name = name.replace(" ", "_")
        return name
    
    except Exception as e:
         raise Exception("ERROR in parsing image name")
     
def img_name_processing(name):
    if 'MP' in name:
        parts = name.split('-')
        if 'Deconvolved' in name and len(parts) >= 3:
            name = parts[0] + '_' + parts[2]
        elif len(parts) >= 2:
            name = parts[1]
            
        else:
            name = os.path.splitext(name)[0]
            
    name = name.strip().replace(' ', '_').replace(',', '')
    return re.sub(r'[\\/:*?"<>|]', '_', name)

def safe_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', str(name)).strip().rstrip('. ') or 'ROI'
     
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

def save_rois_zip(rois, zip_path):
    """Write every selected ROI to a ZIP, independent of ROI Manager selection."""
    if not rois:
        raise ValueError('Cannot save an empty ROI collection')
    stream = ZipOutputStream(FileOutputStream(zip_path))
    try:
        for i, roi in enumerate(rois):
            name = safe_filename(roi.getName() or 'ROI')
            entry_name = '%03d_%s.roi' % (i + 1, name)
            data = ByteArrayOutputStream()
            RoiEncoder(data).write(roi)
            stream.putNextEntry(ZipEntry(entry_name))
            stream.write(data.toByteArray())
            stream.closeEntry()
    finally:
        stream.close()

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


def close_all_csv_tables():
    windows = WindowManager.getAllNonImageWindows()
    if windows is not None:
        for w in windows:
            title = w.getTitle()
            if title.endswith(".csv"):
                w.dispose()

def cleanup_iteration():
    rm = RoiManager.getInstance()
    if rm is not None:
        rm.reset()
        rm.close()
    
def semi_manual_img_process(imp, output_dir, p):
    # Parameters:
    DAPI_CHANNEL =  p['DAPI_CHANNEL'] - 1
    MEASURE_CHANNEL =  p['MEASURE_CHANNEL'] - 1
    
    if max(DAPI_CHANNEL, MEASURE_CHANNEL) >= imp.getNChannels():
        raise ValueError('Image has %d channel(s); requested DAPI=%d, measurement=%d' %
                         (imp.getNChannels(), p['DAPI_CHANNEL'], p['MEASURE_CHANNEL']))
    
    img_title = img_name_processing(imp.getTitle())
    
    split_imps = []
    rm = None
    
    try:
        # ChannelSplitter returns channels in C1, C2, ... order.
        split_imps = list(ChannelSplitter.split(imp))
        dapi_imp = split_imps[DAPI_CHANNEL]
        meas_imp = split_imps[MEASURE_CHANNEL]
        for channel in split_imps:
             channel.show()
             channel.getProcessor().resetMinAndMax()
             IJ.run(channel, 'Enhance Contrast', 'saturated=0.35')
             channel.updateAndDraw()
        
        rm = ensure_roi_manager(reset=True) # clean roi manager before launch
        
        while True:
            gd = NonBlockingGenericDialog('Select ROIs: ' + img_title)
            gd.addMessage('Draw ROI(s) on the DAPI image and click Add in ROI Manager.\n'
                          'Click OK to continue, or Cancel to stop processing.')
            gd.showDialog()
            if gd.wasCanceled():
                return False
            rois = list(rm.getRoisAsArray()) # list of ROIs in roi manager
            if not rois:
                WaitForUserDialog('ROI Warning', 'Add at least one ROI to ROI Manager.').show()
                continue
            if p['one_roi'] and len(rois) != 1:
                WaitForUserDialog('ROI Warning', 'Exactly one ROI is required. Please adjust ROI Manager.').show()
                continue
            break
            
        # Export each ROI without changing the original measurement image.
        for i, roi in enumerate(rois):
            roi_name = roi.getName()
            if not roi_name or not roi_name.strip():
                roi_name = "ROI_%02d" % (i + 1)
            roi_name = safe_filename(roi_name)
            work = None
            try:
                work = meas_imp.duplicate()
                work.setTitle('MEAS_work')
                work.show()
                work.setRoi(roi.clone())
                IJ.run(work, 'Clear Outside', '')
                work.killRoi()
                outname = 'C%d_%s_ROI_%02d_%s.tif' % (
                        p['MEASURE_CHANNEL'], img_title, i + 1, roi_name)
                IJ.saveAs(work, 'Tiff', os.path.join(output_dir, outname))
                
            finally:
                if work is not None:
                    work.changes = False
                    work.close()
                imp.changes = False
                imp.close()
            
        roi_path = os.path.join(output_dir, 'C%d_%s_rois.zip' % (p['DAPI_CHANNEL'], img_title))
        save_rois_zip(rois, roi_path)
        IJ.log('Saved %d ROIs to %s' % (len(rois), roi_path))
        return True
    
    finally:
        close_images(split_imps)
        if rm is not None:
            rm.reset()
                 

#----------------------------
# MAIN
#----------------------------
def main():
    # Check if at least one image is opened
    ids = WindowManager.getIDList() or []
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
        IJ.error('No suitable images are open.')
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
    for i, imp in enumerate(images):
        IJ.log('Processing %d/%d: %s' % (i + 1, len(images), imp.getTitle()))
        try:
            if not semi_manual_img_process(imp, output_dir, params):
                IJ.log('Cancelled by user; stopping batch.')
                
            IJ.log('Successfully saved: ' + imp.getTitle())

        except Exception as e:
            # log immediately
            IJ.log("ERROR in {}: {}".format(imp.getTitle(), e))
            IJ.log(traceback.format_exc())  # comment out if too verbose

# Run program
if __name__ == "__main__":
    main()
