import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import f, chi2
from scipy.spatial.distance import pdist, squareform, mahalanobis

from skbio import DistanceMatrix
from skbio.stats.distance import permanova, permdisp


def plot_pca_centroids_ci(
    pca_df,
    group_col="sample",
    confidence=0.95,
    figsize=(9, 7)
):
    fig, ax = plt.subplots(figsize=figsize)

    groups = pca_df[group_col].dropna().unique()
    colors = sns.color_palette("Set2", len(groups))

    centroids = {}

    for group, color in zip(groups, colors):

        data = pca_df.loc[
            pca_df[group_col] == group,
            ["PC1", "PC2"]
        ].dropna()

        X = data.to_numpy()
        n = len(X)

        if n < 3:
            continue

        # Centroid
        centroid = X.mean(axis=0)
        centroids[group] = centroid

        # Covariance matrix
        cov = np.cov(X, rowvar=False)

        # 95% confidence ellipse for the mean
        p = 2

        scale = (
            p * (n - 1) / (n - p)
            * f.ppf(confidence, p, n - p)
        )

        # Covariance of the sample mean
        cov_mean = cov / n

        eigenvalues, eigenvectors = np.linalg.eigh(cov_mean)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        width, height = 2 * np.sqrt(
            np.maximum(eigenvalues, 0) * scale
        )

        angle = np.degrees(
            np.arctan2(
                eigenvectors[1, 0],
                eigenvectors[0, 0]
            )
        )

        # Plot observations
        ax.scatter(
            X[:, 0],
            X[:, 1],
            color=color,
            alpha=0.3,
            s=35,
            label=f"{group} (n={n})"
        )

        # Plot centroid
        ax.scatter(
            centroid[0],
            centroid[1],
            color=color,
            marker="X",
            s=250,
            edgecolor="black",
            linewidth=1.2,
            zorder=5
        )

        # Plot confidence ellipse
        ellipse = Ellipse(
            xy=centroid,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=0.2
        )

        ax.add_patch(ellipse)

        border = Ellipse(
            xy=centroid,
            width=width,
            height=height,
            angle=angle,
            facecolor="none",
            edgecolor=color,
            linewidth=2
        )

        ax.add_patch(border)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA — Centroids with {confidence:.0%} Confidence Ellipses")

    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.show()

    return centroids


def _bootstrap_pca_centroids(
    pca_df,
    group_col="sample",
    features=("PC1", "PC2"),
    n_boot=5000,
    confidence=0.95,
    random_state=42,
    figsize=(9, 7),
    show_points=True,
    pca=None
):
    rng = np.random.default_rng(random_state)
    
    # Validate inputs
    if len(features) != 2:
        raise ValueError("Exactly two PCA features are required.")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    if n_boot < 2:
        raise ValueError("n_boot must be at least 2.")
    
    required = [group_col, *features]
    missing = [col for col in required if col not in pca_df.columns]

    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # Make a figure
    fig, ax = plt.subplots(figsize=figsize)

    groups = pca_df[group_col].dropna().unique()
    colors = sns.color_palette("Set2", len(groups))

    results = {}
    scale = chi2.ppf(confidence, df=2)

    for group, color in zip(groups, colors):

        data = pca_df.loc[
            pca_df[group_col] == group,
            list(features)
        ].dropna()

        X = data.to_numpy(dtype=float)
        n = len(X)

        if n < 3:
            print(f"Skipping {group}: only {n} observations.")
            continue

        # Original centroid
        centroid = X.mean(axis=0)

        # Bootstrap resampling
        indices = rng.integers(
            0, n,
            size=(n_boot, n)
        )

        bootstrap_centroids = X[indices].mean(axis=1)

        # Bootstrap covariance of centroid
        cov = np.cov(bootstrap_centroids, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Ellipse dimensions
        width, height = 2 * np.sqrt(
            np.maximum(eigenvalues, 0) * scale
        )

        angle = np.degrees(
            np.arctan2(
                eigenvectors[1, 0],
                eigenvectors[0, 0]
            )
        )

        # Original observations
        if show_points:
            ax.scatter(
                X[:, 0],
                X[:, 1],
                color=color,
                alpha=0.25,
                s=30,
                label=f"{group} (n={n})"
            )

        # Bootstrap centroid distribution
        ax.scatter(
            bootstrap_centroids[:, 0],
            bootstrap_centroids[:, 1],
            color=color,
            alpha=0.025,
            s=8
        )

        # Original centroid
        ax.scatter(
            centroid[0],
            centroid[1],
            color=color,
            marker="X",
            s=250,
            edgecolor="black",
            linewidth=1.2,
            zorder=5
        )

        # Confidence ellipse
        ellipse = Ellipse(
            xy=centroid,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=0.15
        )

        ax.add_patch(ellipse)

        border = Ellipse(
            xy=centroid,
            width=width,
            height=height,
            angle=angle,
            facecolor="none",
            edgecolor=color,
            linewidth=2
        )

        ax.add_patch(border)

        # Marginal bootstrap confidence intervals
        ci_low = np.quantile(
            bootstrap_centroids,
            (1 - confidence) / 2,
            axis=0
        )

        ci_high = np.quantile(
            bootstrap_centroids,
            1 - (1 - confidence) / 2,
            axis=0
        )

        results[group] = {
            "centroid": centroid,
            "CI_low": ci_low,
            "CI_high": ci_high,
            "bootstrap_centroids": bootstrap_centroids
        }

    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

    # PCA axis labels
    if pca is not None:
        variance = pca.explained_variance_ratio_

        if len(variance) < 2:
            raise ValueError("PCA must contain at least two components.")

        ax.set_xlabel(f"PC1 ({variance[0] * 100:.2f}%)")
        ax.set_ylabel(f"PC2 ({variance[1] * 100:.2f}%)")
    else:
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])

    ax.set_title(
        f"PCA Centroids — {confidence:.0%} Bootstrap Confidence Ellipses"
    )

    if results:
        ax.legend(frameon=False)
        
    ax.grid(alpha=0.15)
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.show()

    return results


def bootstrap_pca_centroids(
    pca_df,
    group_col="sample",
    features=("PC1", "PC2"),
    n_boot=5000,
    confidence=0.95,
    random_state=42,
    figsize=(9, 7),
    show_points=True,
    show_bootstrap=False,
    pca=None,
    palette="Set2",
    centroid_size=85
):
    rng = np.random.default_rng(random_state)

    # Validate inputs
    if len(features) != 2:
        raise ValueError("Exactly two PCA features are required.")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    if n_boot < 2:
        raise ValueError("n_boot must be at least 2.")

    required = [group_col, *features]
    missing = [c for c in required if c not in pca_df.columns]

    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # Plot style
    sns.set_theme(style="white", context="paper", font_scale=1.3)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    groups = pca_df[group_col].dropna().unique()
    colors = sns.color_palette(palette, len(groups))
    color_map = dict(zip(groups, colors))

    results = {}
    scale = chi2.ppf(confidence, df=2)

    for group in groups:

        color = color_map[group]

        data = pca_df.loc[
            pca_df[group_col] == group,
            list(features)
        ].dropna()

        X = data.to_numpy(dtype=float)
        n = len(X)

        if n < 3:
            print(f"Skipping {group}: only {n} observations.")
            continue

        # Original centroid
        centroid = X.mean(axis=0)

        # Bootstrap resampling
        indices = rng.integers(
            0, n, size=(n_boot, n)
        )

        bootstrap_centroids = X[indices].mean(axis=1)

        # Covariance of bootstrap centroids
        cov = np.cov(bootstrap_centroids, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0)
        eigenvectors = eigenvectors[:, order]

        # Confidence ellipse dimensions
        width, height = 2 * np.sqrt(eigenvalues * scale)

        angle = np.degrees(
            np.arctan2(
                eigenvectors[1, 0],
                eigenvectors[0, 0]
            )
        )

        # Original observations
        if show_points:
            ax.scatter(
                X[:, 0],
                X[:, 1],
                color=color,
                s=32,
                alpha=0.40,
                edgecolors="none",
                label=f"{group} (n={n})",
                zorder=2
            )

        # Optional bootstrap centroid cloud
        if show_bootstrap:
            ax.scatter(
                bootstrap_centroids[:, 0],
                bootstrap_centroids[:, 1],
                color=color,
                alpha=0.035,
                s=7,
                edgecolors="none",
                zorder=3
            )

        # Confidence ellipse
        ellipse = Ellipse(
            xy=centroid,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            linewidth=1.7,
            alpha=0.20,
            zorder=3
        )
        ax.add_patch(ellipse)

        # Clear ellipse border
        border = Ellipse(
            xy=centroid,
            width=width,
            height=height,
            angle=angle,
            facecolor="none",
            edgecolor=color,
            linewidth=1.6,
            zorder=4
        )
        ax.add_patch(border)

        # Smaller centroid, same color as observations
        ax.scatter(
            centroid[0],
            centroid[1],
            color=color,
            marker="o",
            s=centroid_size,
            edgecolor="white",
            linewidth=0.8,
            zorder=6
        )

        # Marginal confidence intervals
        ci_low = np.quantile(
            bootstrap_centroids,
            (1 - confidence) / 2,
            axis=0
        )

        ci_high = np.quantile(
            bootstrap_centroids,
            1 - (1 - confidence) / 2,
            axis=0
        )

        results[group] = {
            "centroid": centroid,
            "CI_low": ci_low,
            "CI_high": ci_high,
            "bootstrap_centroids": bootstrap_centroids
        }

    # Reference axes
    ax.axhline(
        0, color="gray", linestyle="--",
        linewidth=0.8, alpha=0.35, zorder=1
    )
    ax.axvline(
        0, color="gray", linestyle="--",
        linewidth=0.8, alpha=0.35, zorder=1
    )

    # PCA axis labels
    if pca is not None:
        variance = pca.explained_variance_ratio_

        if len(variance) < 2:
            raise ValueError(
                "PCA must contain at least two components."
            )

        ax.set_xlabel(
            f"PC1 ({variance[0] * 100:.1f}%)",
            fontsize=12, fontweight="normal"
        )
        ax.set_ylabel(
            f"PC2 ({variance[1] * 100:.1f}%)",
            fontsize=12, fontweight="normal"
        )
    else:
        ax.set_xlabel(features[0], fontsize=12)
        ax.set_ylabel(features[1], fontsize=12)

    ax.set_title(
        f"PCA — {confidence:.0%} Confidence Ellipses",
        fontsize=14,
        fontweight="bold",
        pad=18
    )

    # Legend
    if results and show_points:
        ax.legend(
            frameon=False,
            fontsize=10,
            loc="best",
            markerscale=1.3
        )

    # Minimal grid
    ax.grid(
        True,
        linestyle=":",
        linewidth=0.7,
        alpha=0.18
    )

    sns.despine(ax=ax)

    ax.tick_params(
        axis="both",
        labelsize=10,
        length=0
    )

    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.show()

    return results


def pca_permanova(
    pca_df,
    group_col="sample",
    features=("PC1", "PC2"),
    permutations=9999
):
    # Prepare data
    df = pca_df[
        [*features, group_col]
    ].dropna().copy()

    X = df[list(features)].to_numpy()
    groups = df[group_col].to_numpy()

    # Euclidean distance matrix
    dm = DistanceMatrix(
        squareform(pdist(X, metric="euclidean"))
    )

    # Statistical tests
    result_permanova = permanova(
        dm,
        grouping=groups,
        permutations=permutations,
        seed=42
    )

    result_permdisp = permdisp(
        dm,
        grouping=groups,
        permutations=permutations,
        seed=42
    )

    # Calculate PERMANOVA R²
    F = result_permanova["test statistic"]
    n = result_permanova["sample size"]
    k = result_permanova["number of groups"]

    r_squared = (F * (k - 1)) / (F * (k - 1) + (n - k))

    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            "Test": "PERMANOVA",
            "Statistic": F,
            "p_value": result_permanova["p-value"],
            "R2": r_squared,
            "n": n,
            "groups": k
        },
        {
            "Test": "PERMDISP",
            "Statistic": result_permdisp["test statistic"],
            "p_value": result_permdisp["p-value"],
            "R2": None,
            "n": result_permdisp["sample size"],
            "groups": result_permdisp["number of groups"]
        }
    ])

    return results_df

def detect_pca_outliers(
    df,
    group_col="sample",
    features=("PC1", "PC2"),
    confidence=0.99
):
    result = df.copy()

    result["mahalanobis_distance"] = np.nan
    result["outlier"] = False

    threshold = chi2.ppf(confidence, df=len(features))

    for group, group_df in result.groupby(group_col):

        X = group_df[list(features)].dropna()

        if len(X) <= len(features):
            continue

        # Group centroid
        centroid = X.mean(axis=0).values

        # Covariance matrix
        cov = np.cov(X.values, rowvar=False)

        # Inverse covariance
        inv_cov = np.linalg.pinv(cov)

        # Mahalanobis distances
        distances = np.array([
            mahalanobis(row, centroid, inv_cov)
            for row in X.values
        ])

        result.loc[X.index, "mahalanobis_distance"] = distances

        result.loc[X.index, "outlier"] = (
            distances**2 > threshold
        )

    return result