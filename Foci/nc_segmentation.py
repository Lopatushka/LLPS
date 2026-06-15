from ij import IJ, WindowManager
from ij.gui import WaitForUserDialog
from ij.plugin.frame import RoiManager
from ij.measure import ResultsTable
from ij.plugin import Duplicator
from ij.process import ImageProcessor
import os

def get_roi_manager():
    rm = RoiManager.getInstance()
    if rm is None:
        rm = RoiManager()
    rm.reset()
    return rm

def draw_nuclei(rm):
    IJ.setTool("freehand")

    WaitForUserDialog(
        "Nuclei Segmentation",
        "Draw nuclei ROIs.\n"
        "Press 't' after each ROI to add it to ROI Manager.\n"
        "Click OK when finished."
    ).show()

    n_nuclei = rm.getCount()

    for i in range(n_nuclei):
        rm.select(i)
        rm.rename("Nucleus_%d" % (i + 1))

    return n_nuclei

def draw_cells(rm, n_nuclei):
    WaitForUserDialog(
        "Cell Segmentation",
        "Draw whole-cell ROIs.\n"
        "Press 't' after each ROI.\n"
        "Click OK when finished."
    ).show()

    total_rois = rm.getCount()
    n_cells = total_rois - n_nuclei

    for i in range(n_cells):
        rm.select(n_nuclei + i)
        rm.rename("Cell_%d" % (i + 1))

    return n_cells

def create_mask(width, height, title):
    return IJ.createImage(title, "8-bit black", width, height, 1)


def fill_mask(mask, rois):
    for roi in rois:
        mask.setRoi(roi)
        IJ.run(mask, "Fill", "slice")
        
def build_masks(rm, n_nuclei, n_cells, width, height):
    nuclei_mask = create_mask(width, height, "Nuclei_mask")
    cell_mask = create_mask(width, height, "Cell_mask")

    nuclei_rois = [rm.getRoi(i) for i in range(n_nuclei)]
    cell_rois = [rm.getRoi(n_nuclei + i) for i in range(n_cells)]

    fill_mask(nuclei_mask, nuclei_rois)
    fill_mask(cell_mask, cell_rois)

    return nuclei_mask, cell_mask


def create_cytoplasm_mask(cell_mask, nuclei_mask):
    cyto_mask = cell_mask.duplicate()
    cyto_mask.setTitle("Cytoplasm_mask")

    IJ.run(
        cyto_mask,
        "Image Calculator...",
        "image1=Cytoplasm_mask operation=Subtract image2=Nuclei_mask create"
    )

    result = WindowManager.getImage("Result of Cytoplasm_mask")

    if result:
        result.setTitle("Cytoplasm_mask")

    return result

def main():

    imp = IJ.getImage()

    if imp is None:
        IJ.showMessage("Error", "No image open.")
        return

    output_dir = IJ.getDirectory("Choose output folder")

    rm = get_roi_manager()

    n_nuclei = draw_nuclei(rm)
    n_cells = draw_cells(rm, n_nuclei)

    nuclei_mask, cell_mask = build_masks(
        rm,
        n_nuclei,
        n_cells,
        imp.getWidth(),
        imp.getHeight()
    )

    cyto_mask = create_cytoplasm_mask(
        cell_mask,
        nuclei_mask
    )

    results_table = measure_rois(
        imp,
        rm,
        n_nuclei,
        n_cells
    )

    if output_dir:
        save_results(
            output_dir,
            imp.getTitle(),
            nuclei_mask,
            cell_mask,
            cyto_mask,
            rm,
            results_table
        )

    IJ.showMessage(
        "Finished",
        "Segmentation completed successfully."
    )


if __name__ == "__main__":
    main()