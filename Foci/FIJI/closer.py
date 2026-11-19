# Close all images that are open in Fiji.
# Asks for confirmation first; unsaved changes are discarded (no "Save?" prompt
# for every image). Tables, the Log and other non-image windows stay open.
from ij import IJ, WindowManager
from ij.gui import GenericDialog


def main():
    ids = WindowManager.getIDList()  # IDs of all open images (None if none)
    if not ids:
        IJ.log("No images open.")
        return

    gd = GenericDialog("Close all images")
    gd.addMessage("Close all {} open image(s)?\n"
                  "Unsaved changes will be lost.".format(len(ids)))
    gd.showDialog()
    if gd.wasCanceled():
        return

    closed = 0
    for wid in ids:
        imp = WindowManager.getImage(wid)
        if imp is None:  # already closed
            continue
        imp.changes = False  # do not ask to save
        imp.close()
        closed += 1

    IJ.log("Closed {} image(s).".format(closed))


main()
