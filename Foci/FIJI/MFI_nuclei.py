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
    
    # Ask user where to save outputs
    output_dir = IJ.getDirectory("Choose a directory to save data")
    if output_dir is None:
        IJ.error("No output directory is selected!")
        return
    
    # --- Measurements of ROIs ---
    # Create an empty results table
    rt = ResultsTable()
    rt.show("Results")
    
    # ---- Loop: show GUI per image, then process ----
    for call_id, imp in enumerate(unique_images, start=1):
        # Make Log message
        msg = "Processing {}/{}: {}".format(call_id, n, imp.getTitle())
        IJ.log(msg)

        try:
            # Make measurememts on the WORK image
            stats = imp.getStatistics(
            Measurements.AREA | Measurements.MEAN
            )
        
            # Fill the table with results
            rt.incrementCounter()
            rt.addValue("Filename", imp.getTitle())
            rt.addValue("Area", stats.area)
            rt.addValue("Mean", stats.mean)
            
                
        except Exception as e:
            # log immediately
            IJ.log("ERROR in {}: {}".format(imp.getTitle(), e))
            IJ.log(traceback.format_exc())  # comment out if too verbose
            continue
    
    # Save Results as CSV
    table_name = "MFI_nuclei.csv"
    results_path = os.path.join(output_dir, table_name)
    IJ.saveAs("Results", results_path)
       
    close_all_csv_tables()
    close_images(images)
        
if __name__ == "__main__":
    main()