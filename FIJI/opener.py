# Open all files in a folder whose name contains a given pattern and has a given
# extension (case-insensitive). Optionally searches subfolders as well.
# Files are opened in alphabetical order; files that cannot be opened are skipped
# and listed in the Log window.
from ij import IJ
from ij.gui import GenericDialog
import os


def find_files(root, pattern, ext, recursive):
    """Sorted full paths of the files in root (and subfolders if recursive) whose
    lowercased name ends with ext and contains pattern (empty pattern = any name)."""
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort(key=lambda d: d.lower())  # walk subfolders in alphabetical order
        for name in sorted(filenames, key=lambda f: f.lower()):
            lower = name.lower()  # compare in lower case, but keep the real name for the path
            if lower.endswith(ext) and pattern in lower:
                paths.append(os.path.join(dirpath, name))
        if not recursive:
            break  # only the top folder
    return paths


def main():
    # Choose the directory to open files from (None if Cancel is pressed)
    root = IJ.getDirectory("Choose a root directory")
    if not root:
        return

    # Choose the pattern and extension to filter files
    gd = GenericDialog("Open files by pattern")
    gd.addStringField("Filename contains (empty = any):", "ROI", 15)
    gd.addStringField("Extension (e.g. .tif):", ".tif", 10)
    gd.addCheckbox("Search subfolders", False)

    gd.showDialog()
    if gd.wasCanceled():
        return

    pattern = gd.getNextString().strip().lower()
    ext = gd.getNextString().strip().lower()
    recursive = gd.getNextBoolean()

    # --- Validate input ---
    if ext == "":
        IJ.error("Extension cannot be empty!")
        return
    if not ext.startswith("."):  # auto-add dot if missing
        ext = "." + ext

    paths = find_files(root, pattern, ext, recursive)
    if not paths:
        IJ.log("No files matching '*{}*{}' found in: {}".format(pattern, ext, root))
        return

    opened = 0
    for path in paths:
        imp = IJ.openImage(path)  # None if the file cannot be read as an image
        if imp is None:
            IJ.log("Could not open, skipped: {}".format(path))
            continue
        imp.show()
        opened += 1

    IJ.log("Opened {} of {} file(s) from: {}".format(opened, len(paths), root))


main()
