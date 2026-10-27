import pandas as pd
import numpy as np

import umap

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.cluster import KMeans

from scipy.stats import chi2, spearmanr
from scipy.spatial.distance import pdist, squareform

from skbio import DistanceMatrix
from skbio.stats.distance import permanova, permdisp, DistanceMatrix

from statsmodels.stats.multitest import multipletests

def confidence_ellipse(x, y, ax, confidence=0.95, **kwargs):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove NaN and infinite values
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    # At least 3 points required
    if len(x) < 3:
        return

    # Covariance matrix
    cov = np.cov(x, y)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort in descending order
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    eigenvectors = eigenvectors[:, order]

    # Rotation angle
    angle = np.degrees(
        np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    )

    # Confidence scaling
    scale = np.sqrt(chi2.ppf(confidence, df=2))

    width, height = 2 * scale * np.sqrt(eigenvalues)

    # Default ellipse appearance
    kwargs.setdefault("fill", False)
    kwargs.setdefault("linewidth", 2)

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        **kwargs
    )

    ax.add_patch(ellipse)