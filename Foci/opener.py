# Open files from a folder if filename contains "mask" and has a given extension.
from ij import IJ
from ij.gui import GenericDialog
import os

def main():
    # Chose the directory to open files from
    root = IJ.getDirectory("Choose a root directory")
    if not root:
        IJ.error("No directory selected!")
        return

    # chose the pattern and extension to filter files
    gd = GenericDialog("Open files by pattern")

    gd.addStringField("Filename contains:", "ROI", 15)
    gd.addStringField("Extension (e.g. .tif):", ".tif", 10)

    gd.showDialog()
    if gd.wasCanceled():
        IJ.error("Cancelling.")
        return

    # Get variables
    pattern = gd.getNextString().strip()
    ext = gd.getNextString().strip()

    # --- Validate input ---
    if pattern == "":
        IJ.error("Pattern cannot be empty!")
        return

    if ext == "":
        IJ.error("Extension cannot be empty!")
        return

    # auto-add dot if missing
    if not ext.startswith("."):
        ext = "." + ext

    pattern = pattern.lower()
    ext = ext.lower()

    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            name = name.lower()
            if name.endswith(ext.lower()) and (pattern in name):
                path = os.path.join(dirpath, name)
                IJ.openImage(path).show()
                count += 1

    IJ.log("Opened {} file(s) from: {}".format(count, root))

main()