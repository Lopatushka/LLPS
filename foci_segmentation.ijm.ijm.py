# Libraries import
from ij import IJ
from ij.plugin.frame import RoiManager
from ij import WindowManager
import os
import csv

# Choose folder to select results
output_dir = IJ.getDirectory("Choose a directory!")

# Activate the currently opened image
imp = IJ.getImage()
image_name = imp.getTitle() # get image name

# Run ROI manager. Extract all ROIs
roi_manager = RoiManager.getInstance()
if roi_manager is None or roi_manager.getCount() == 0:
    IJ.error("No ROIs in ROI Manager.")
    
rois = roi_manager.getRoisAsArray()

# Cycle through the all ROIs of the image
for i, roi in enumerate(rois):
	roi_name = roi.getName()
	if roi_name is None or roi_name.strip() == "":
		roi_name = "ROI_{}".format(i + 1)
		
	# Set ROI and crop image
	imp.setRoi(roi) # activate the current ROI
	dup = imp.crop() # crop image
	dup = dup.duplicate() # duplicate thi ROI to a new image
	dup.setTitle("ROI_{}".format(i + 1)) # set name to this ROI
	dup_name = dup.getTitle() # get duplicated image name name
	
	# Image processing
	#IJ.run(dup, "Subtract Background...", "rolling=4") # sybstact background 
	
	dup.show() # show the image 
	
	# Run ThunderSTORM plugin
	# threshold=3*std(Wave.F1)!
	try:
		IJ.run(dup,
		  "Run analysis",
		  "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 "
		  "detector=[Local maximum] connectivity=8-neighbourhood threshold=3*std(Wave.F1) "
		  "estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 "
		  "method=[Weighted Least squares] full_image_fitting=false mfaenabled=false "
		  "renderer=[No Renderer]")
		
		# Export results in the .csv file
		table_name = "{}_{}.csv".format(image_name[:-4], dup_name)
		csv_path = os.path.join(output_dir, table_name)
		IJ.run(
		    "Export results",
		    "filepath={} fileformat=[CSV (comma separated)] "
		    "sigma=false intensity=true chi2=false offset=false "
		    "saveprotocol=false x=true y=true bkgstd=false "
		    "id=true uncertainty=false frame=false".format(csv_path))
		    
		# Save cropped image
		file_name = "{}_{}.png".format(image_name[:-4], dup_name)
		image_path = os.path.join(output_dir, file_name)
		IJ.save(dup, image_path)
		
		# Close cropped image
		dup.close()
		
		# Close table with results
		windows = WindowManager.getNonImageWindows()
		for w in windows:
			if w.getTitle() == "ThunderSTORM: results":
				w.dispose()


	except Exception as e:
            IJ.log("Error running ThunderSTORM analysis for '{}': {}".format(roi_name, str(e)))

	