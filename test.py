from ij import IJ, WindowManager
from ij.gui import GenericDialog
from ij.plugin.frame import RoiManager
from ij.measure import Measurements, ResultsTable
from ij.plugin.filter import ParticleAnalyzer
from ij.plugin.filter import BackgroundSubtracter
from ij.process import AutoThresholder
from ij.io import RoiEncoder
import os
import csv
import traceback

def subtract_background(imp, radius, light_background=False, use_paraboloid=False, do_presmooth=True):
    radius = float(radius)
    ip = imp.getProcessor()  # ImageProcessor of current slice
    BackgroundSubtracter().rollingBallBackground(
        ip,
        radius,
        False,
        bool(light_background),
        bool(use_paraboloid),
        bool(do_presmooth),
        False
    )
    imp.updateAndDraw()

imp = IJ.getImage()  # gets active image or throws error
img_title = imp.getTitle()
print(img_title)

subtract_background(imp, 10, light_background=False, use_paraboloid=False, do_presmooth=True)
