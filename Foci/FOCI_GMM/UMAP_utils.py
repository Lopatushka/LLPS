import pandas as pd
import numpy as np
import math
import umap

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.cluster import KMeans

from scipy.stats import chi2, spearmanr
from scipy.spatial.distance import pdist, squareform, cdist

from skbio import DistanceMatrix
from skbio.stats.distance import permanova, permdisp, DistanceMatrix

from statsmodels.stats.multitest import multipletests

def calculate_umap(df, features, n_neighbors=20):

    X = df[features].copy()

    X_scaled = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        n_jobs=1
    )

    embedding = reducer.fit_transform(X_scaled)

    result = df.copy()

    result["UMAP1"] = embedding[:, 0]
    result["UMAP2"] = embedding[:, 1]

    return result

def plot_cluster_centroids_ci(
    umap_df,
    cluster_col="cluster",
    group_col="sample",
    features=("UMAP1", "UMAP2"),
    n_boot=5000,
    confidence=0.95,
    random_state=42,
    show_points=True,
    figsize=(14, 11),
    point_size=55,
    centroid_size=300,
    zoom_padding=0.15
):

    rng = np.random.default_rng(random_state)

    df = umap_df.dropna(
        subset=[cluster_col, group_col, *features]
    ).copy()

    clusters = sorted(df[cluster_col].unique())
    groups = sorted(df[group_col].unique())

    # Consistent colors across clusters
    palette = dict(zip(
        groups,
        sns.color_palette("Set2", len(groups))
    ))

    centroid_results = []

    # ONE LARGE FIGURE PER CLUSTER
    for cluster in clusters:

        cluster_df = df[df[cluster_col] == cluster]

        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=120
        )

        for group, subset in cluster_df.groupby(group_col):

            X = subset[list(features)].to_numpy()
            n = len(X)

            if n == 0:
                continue

            color = palette[group]

            # ---------------------------------
            # 1. Individual cells
            # ---------------------------------
            if show_points:
                ax.scatter(
                    X[:, 0],
                    X[:, 1],
                    s=point_size,
                    alpha=0.30,
                    color=color,
                    edgecolors="none",
                    zorder=1
                )

            # ---------------------------------
            # 2. Observed centroid
            # ---------------------------------
            centroid = X.mean(axis=0)

            # ---------------------------------
            # 3. Bootstrap 95% CI ellipse
            # ---------------------------------
            if n >= 3:

                boot_idx = rng.integers(
                    0,
                    n,
                    size=(n_boot, n)
                )

                boot_centroids = X[boot_idx].mean(axis=1)

                cov = np.cov(
                    boot_centroids,
                    rowvar=False
                )

                eigenvalues, eigenvectors = np.linalg.eigh(cov)

                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]

                scale = np.sqrt(
                    chi2.ppf(confidence, df=2)
                )

                width, height = (
                    2 * scale *
                    np.sqrt(np.maximum(eigenvalues, 0))
                )

                angle = np.degrees(
                    np.arctan2(
                        eigenvectors[1, 0],
                        eigenvectors[0, 0]
                    )
                )

                # Filled confidence ellipse
                ellipse = Ellipse(
                    xy=centroid,
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.18,
                    linewidth=2
                )

                ax.add_patch(ellipse)

                # Confidence ellipse outline
                outline = Ellipse(
                    xy=centroid,
                    width=width,
                    height=height,
                    angle=angle,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.5,
                    zorder=3
                )

                ax.add_patch(outline)

            # ---------------------------------
            # 4. Centroid marker
            # ---------------------------------
            ax.scatter(
                centroid[0],
                centroid[1],
                s=centroid_size,
                marker="X",
                color=color,
                edgecolor="black",
                linewidth=1.2,
                zorder=5,
                label=f"{group} (n={n})"
            )

            centroid_results.append({
                "cluster": cluster,
                "sample": group,
                "n": n,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1]
            })

        # ---------------------------------
        # 5. Formatting
        # ---------------------------------
        ax.set_title(
            f"Cluster {cluster} — 95% CI of centroids",
            fontsize=22,
            fontweight="bold",
            pad=20
        )

        ax.set_xlabel(features[0], fontsize=18)
        ax.set_ylabel(features[1], fontsize=18)

        ax.tick_params(labelsize=15)

        ax.legend(
            title="Mutant",
            fontsize=14,
            title_fontsize=15,
            loc="best",
            frameon=True
        )

        ax.grid(alpha=0.15)

        ax.spines[["top", "right"]].set_visible(False)

        # ---------------------------------
        # 6. Zoom to individual cluster
        # ---------------------------------
        x = cluster_df[features[0]]
        y = cluster_df[features[1]]

        x_range = x.max() - x.min()
        y_range = y.max() - y.min()

        x_pad = max(x_range * zoom_padding, 0.1)
        y_pad = max(y_range * zoom_padding, 0.1)

        ax.set_xlim(
            x.min() - x_pad,
            x.max() + x_pad
        )

        ax.set_ylim(
            y.min() - y_pad,
            y.max() + y_pad
        )

        plt.tight_layout()
        plt.show()

    return pd.DataFrame(centroid_results)

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