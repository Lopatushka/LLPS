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
    microscope_type = ["Airyscan", "NSpark"]
    gd.addChoice("Microscope type:", microscope_type, "NSpark")
    
    gd.addNumericField("DAPI channel (1-based):", 1, 0)
    gd.addNumericField("Measurement channel (1-based):", 2, 0)
    gd.addCheckbox('Exactly one ROI per image', False)
    
    gd.showDialog()
    if gd.wasCanceled():
        return None
    
    microscope = gd.getNextChoice()
    dapi = int(gd.getNextNumber())
    measure = int(gd.getNextNumber())
    one_roi = gd.getNextBoolean()
    
    if dapi < 1 or measure < 1:
        IJ.error('Channel numbers must be at least 1.')
        return None
    
    return {'microscope_type': microscope, 'DAPI_CHANNEL': dapi, 'MEASURE_CHANNEL': measure, 'one_roi': one_roi}


def NSpark_name_processing(name):
    try:
        name = str(name) # converts the input to a string first
        name = name.strip() # removes spaces from both ends
        name = name.rstrip('. ') # removes dots and spaces from the end only
        name = name.replace(" ", "_")
        name = name.split(".nd2")[0] + name.split(".nd2")[2]
        return name
    except Exception as e:
         raise Exception("ERROR in NSpark parsing image name")
     
def to_text(value):
    """Return value as text without str(), which fails on non-Latin characters in Jython."""
    if isinstance(value, basestring):
        return value
    return unicode(value)

UNSAFE_CHARS = re.compile(r'[^\w-]', re.UNICODE) # \w = letters of any alphabet, digits, _

def Airyscan_name_processing(name):
    original = name
    try:
        name = os.path.splitext(to_text(name))[0] # drop the file extension
        name = name.strip() # removes spaces from both ends
        name = name.rstrip('. ') # removes dots and spaces from the end only
        # anything not a letter (any alphabet, incl. Cyrillic), digit, _ or - becomes _
        name = UNSAFE_CHARS.sub('_', name)
        name = name.strip('_') # drop leading/trailing _
        return name or 'image'
    except Exception as e:
        raise Exception("ERROR in Airyscan parsing image name %r: %s" % (original, e))

def unique_title(title, used_titles):
    """Append _2, _3, ... if this title was already used in the current batch,
    so two images with the same name don't overwrite each other's files."""
    candidate = title
    k = 2
    while candidate in used_titles:
        candidate = '%s_%d' % (title, k)
        k += 1
    used_titles.add(candidate)
    return candidate

def safe_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', to_text(name)).strip().rstrip('. ') or 'ROI'

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

def clear_outside_all_slices(imp, roi):
    """Set every pixel outside the ROI to 0 on every slice of the stack.

    Does not depend on Fiji's background colour, and never shows the
    'Process all slices?' prompt.
    """
    if roi is None or not roi.isArea():
        raise ValueError('ROI "%s" is not an area selection' % (roi.getName() if roi else None))
    stack = imp.getStack()
    for n in range(1, stack.getSize() + 1):
        ip = stack.getProcessor(n)
        ip.setValue(0)
        ip.fillOutside(roi)
    imp.killRoi()
    imp.updateAndDraw()

def close_images(imps):
    for im in imps:
        if im is None:
            continue
        im.changes = False
        im.close()

def semi_manual_img_process(imp, output_dir, p, used_titles):
    # Parameters:
    DAPI_CHANNEL =  p['DAPI_CHANNEL'] - 1
    MEASURE_CHANNEL =  p['MEASURE_CHANNEL'] - 1
    microscope_type = p['microscope_type']
    
    if max(DAPI_CHANNEL, MEASURE_CHANNEL) >= imp.getNChannels():
        raise ValueError('Image has %d channel(s); requested DAPI=%d, measurement=%d' %
                         (imp.getNChannels(), p['DAPI_CHANNEL'], p['MEASURE_CHANNEL']))
    
    img_title = imp.getTitle()
    
    if microscope_type == "NSpark":
        img_title = NSpark_name_processing(img_title)
            
    elif microscope_type == "Airyscan":
        img_title = Airyscan_name_processing(img_title)

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

        dapi_imp.getWindow().toFront() # ROIs are drawn on the DAPI channel
        
        while True:
            gd = NonBlockingGenericDialog('Select ROIs: ' + img_title)
            gd.addMessage('Draw ROI(s) on the DAPI image and click Add in ROI Manager.\n'
                          'OK: save and continue.\n'
                          'Skip image: close this image without saving and go to the next one.\n'
                          'Cancel: stop processing.')
            # Three buttons: OK / Skip image / Cancel
            gd.enableYesNoCancel('OK', 'Skip image')
            gd.showDialog()
            if gd.wasCanceled():
                return 'cancelled'
            if not gd.wasOKed(): # "Skip image" was pressed
                imp.changes = False
                imp.close() # split images are closed in the finally block below
                return 'skipped'
            rois = list(rm.getRoisAsArray()) # list of ROIs in roi manager
            if not rois:
                WaitForUserDialog('ROI Warning', 'Add at least one ROI to ROI Manager.').show()
                continue
            if p['one_roi'] and len(rois) != 1:
                WaitForUserDialog('ROI Warning', 'Exactly one ROI is required. Please adjust ROI Manager.').show()
                continue
            break

        # Reserve the output name only now, so a skipped image doesn't use it up.
        img_title = unique_title(img_title, used_titles)

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
                clear_outside_all_slices(work, roi)
                outname = 'C%d_%s_ROI_%02d_%s.tif' % (
                        p['MEASURE_CHANNEL'], img_title, i + 1, roi_name)
                IJ.saveAs(work, 'Tiff', os.path.join(output_dir, outname))
                
            finally:
                if work is not None:
                    work.changes = False
                    work.close()

        roi_path = os.path.join(output_dir, '%s_rois.zip' % (img_title))
        save_rois_zip(rois, roi_path)
        IJ.log('Saved %d ROIs to %s' % (len(rois), roi_path))

        # Close the original only after every TIFF and the ROI zip saved.
        imp.changes = False
        imp.close()
        return 'saved'
    
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
        # Skip split-channel windows (C1-..., C2-...) and helper images (adjust if needed)
        if re.match(r'^C\d+-', title) or title in ["DAPI_work", "Nuclei_mask_particles_only"]:
            continue

        images.append(imp)

    if not images:
        IJ.error('No suitable images are open.')
        return

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
    used_titles = set() # output names already used in this batch
    for i, imp in enumerate(images):
        IJ.log('Processing %d/%d: %s' % (i + 1, len(images), imp.getTitle()))
        try:
            result = semi_manual_img_process(imp, output_dir, params, used_titles)
            if result == 'cancelled':
                IJ.log('Cancelled by user; stopping batch.')
                break
            elif result == 'skipped':
                IJ.log('Skipped by user (nothing saved): ' + imp.getTitle())
            else:
                IJ.log('Successfully saved: ' + imp.getTitle())

        except Exception as e:
            # log immediately
            IJ.log("ERROR in {}: {}".format(imp.getTitle(), e))
            IJ.log(traceback.format_exc())  # comment out if too verbose

# Run program
if __name__ == "__main__":
    main()
