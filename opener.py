# Open files from a folder if filename contains "mask" and has a given extension.
from ij import IJ
from ij.gui import GenericDialog
import os

# Chose the directory to open files from
root = IJ.getDirectory("Choose a root directory")
if not root:
    IJ.error("No directory selected!")
    raise SystemExit

# chose the pattern and extension to filter files
gd = GenericDialog("Open files by pattern")

gd.addStringField("Filename contains:", "mask", 15)
gd.addStringField("Extension (e.g. .tif):", ".jpg", 10)

gd.showDialog()
if gd.wasCanceled():
    raise SystemExit

pattern = gd.getNextString().strip()
ext = gd.getNextString().strip()

# --- Validate input ---
if pattern == "":
    IJ.error("Pattern cannot be empty!")
    raise SystemExit

if ext == "":
    IJ.error("Extension cannot be empty!")
    raise SystemExit

# auto-add dot if missing
if not ext.startswith("."):
    ext = "." + ext

pattern = pattern.lower()
ext = ext.lower()

count = 0
for dirpath, dirnames, filenames in os.walk(root):
    for name in filenames:
        lower = name.lower()
        if lower.endswith(ext.lower()) and (pattern.lower() in lower):
            path = os.path.join(dirpath, name)
            IJ.openImage(path).show()
            count += 1

IJ.log("Opened {} file(s) from: {}".format(count, root))
