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

def confidence_ellipse(umap_df, x, y, ax, confidence=0.95, **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)

    # Calculate covariance matrix
    cov = np.cov(x, y)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort in descending order
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Ellipse rotation
    angle = np.degrees(
        np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    )

    # 95% confidence scaling
    scale = np.sqrt(chi2.ppf(confidence, df=2))

    width, height = 2 * scale * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=2,
        **kwargs
    )

    ax.add_patch(ellipse)


# Plot
fig, ax = plt.subplots(figsize=(9, 7))

palette = dict(zip(
    umap_df["sample"].unique(),
    sns.color_palette("Set2", umap_df["sample"].nunique())
))

sns.scatterplot(
    data=umap_df,
    x="UMAP1",
    y="UMAP2",
    hue="sample",
    palette=palette,
    s=80,
    alpha=0.7,
    edgecolor="white",
    ax=ax
)

# Add 95% ellipses
for group, data in umap_df.groupby("sample"):

    confidence_ellipse(
        data["UMAP1"],
        data["UMAP2"],
        ax,
        confidence=0.95,
        edgecolor=palette[group]
    )

    # Mark centroid
    ax.scatter(
        data["UMAP1"].mean(),
        data["UMAP2"].mean(),
        marker="X",
        s=180,
        color=palette[group],
        edgecolor="black",
        linewidth=1.2,
        zorder=5
    )

ax.set_xlabel("UMAP 1", fontsize=13)
ax.set_ylabel("UMAP 2", fontsize=13)
ax.set_title("UMAP — WT vs Mutant (95% Ellipses)", fontsize=15)

ax.legend(title="Sample", frameon=False)
sns.despine()
plt.tight_layout()
plt.show()