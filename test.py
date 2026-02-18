from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage.draw import disk
from matplotlib.patches import Circle
from scipy.stats import spearmanr

def key_from_csv(p: Path) -> str:
    """
    C2...nd2_(series_01)_0233-0247.csv  ->  C2...nd2_(series_01)
    """
    name = p.stem  # no .csv
    # remove trailing _####-#### (or similar) if present
    name = re.sub(r"_\d+-\d+$", "", name)
    return name


path = "/mnt/c/Users/Elena/Desktop/Data_processing/test/res"
pathway = Path(str(path).strip())

files = sorted(pathway.glob("*_extent.csv"))
for f in files:
    key = key_from_csv(f)[:-7]
    df = pd.read_csv(f)
    #print(key, df.shape[0])
    df = df[df["mean_intensity"] > 0.07]
    #df["new_col"] = df["mean_intensity"] > 0.07
    print(key, df.shape[0])