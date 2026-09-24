# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA

from itertools import combinations

from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, permanova
from statsmodels.stats.multitest import multipletests

def pca_from_dataframes(
    df_list,
    labels,
    columns,
    figsize=(5, 4.5),
    dpi=300,
    point_size=35,
    alpha=0.75,
    show=True,
):
    """
    PCA from list of dataframes.

    Parameters
    ----------
    df_list : list[pd.DataFrame]
        One dataframe per sample.

    labels : list[str]
        Sample names.

    columns : list[str]
        Numerical features for PCA.

    Returns
    -------
    pca_df
    pca
    """

    dfs = []

    for df, label in zip(df_list, labels):
        tmp = df[columns].copy()
        tmp["sample"] = label
        dfs.append(tmp)

    data = pd.concat(dfs, ignore_index=True)
    
    # Remove rows containing NaN
    data = data.dropna(subset=columns)
    
    # -------------------------
    # PCA
    # -------------------------
    X = data[columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        pcs,
        columns=["PC1", "PC2"]
    )

    pca_df["sample"] = data["sample"].values

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    for label in labels:

        subset = pca_df[
            pca_df["sample"] == label
        ]

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=point_size,
            alpha=alpha,
            label=label,
            edgecolor="white",
            linewidth=0.4
        )
    
    # Zero reference lines
    ax.axhline(
        0,
        linewidth=0.7,
        linestyle="--",
        alpha=0.35
    )

    ax.axvline(
        0,
        linewidth=0.7,
        linestyle="--",
        alpha=0.35
    )
    
    # Axis labels
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        fontsize=11
    )

    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
        fontsize=11
    )
    
    # Tick appearance
    ax.tick_params(
        axis="both",
        labelsize=9,
        direction="out",
        length=4,
        width=0.8
    )
    
    # Remove unnecessary borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    
    # Legend
    ax.legend(
        frameon=False,
        fontsize=9,
        markerscale=1.1,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    if show:
        plt.show()

    return pca_df, pca, fig, ax

def plot_pca_loadings(
    pca,
    feature_names,
    figsize=(4.5, 4),
    dpi=300,
    decimals=2,
    rotation=0,
    show=True
):
    """
    Plot PCA loadings.

    feature_names : dict
        Mapping:
        {
            "Display name": "dataframe_column_name"
        }
    """

    # Original columns used for PCA
    columns = list(feature_names.values())

    # Names displayed on the figure
    display_names = list(feature_names.keys())

    n_components = pca.components_.shape[0]

    if len(columns) != pca.components_.shape[1]:
        raise ValueError(
            "Number of features in feature_names must match "
            "the number of features used for PCA."
        )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=display_names,
        columns=[f"PC{i + 1}" for i in range(n_components)]
    )

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    im = ax.imshow(
        loadings.values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    # PC labels
    ax.set_xticks(range(n_components))
    ax.set_xticklabels(
        loadings.columns,
        fontsize=10
    )

    # Feature labels
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(
        display_names,
        fontsize=10,
        rotation=rotation,
        va="center"
    )

    # Loading values
    for i in range(len(display_names)):
        for j in range(n_components):

            value = loadings.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.{decimals}f}",
                ha="center",
                va="center",
                fontsize=9
            )

    # Colorbar
    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.05,
        pad=0.04
    )

    cbar.set_label("Loading", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.tick_params(
        axis="both",
        length=0
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return loadings, fig, ax

def pca_centroid_distances(
    pca_df,
    reference="WT"
):
    """
    Calculate Euclidean distances between PCA group centroids
    and a reference group.

    Parameters
    ----------
    pca_df : pd.DataFrame
        Must contain PC1, PC2, and sample columns.

    reference : str
        Reference sample. Default is "WT".

    Returns
    -------
    distances : pd.Series
        Distance of each group centroid from the reference centroid.

    centroids : pd.DataFrame
        PC1 and PC2 coordinates of each group centroid.
    """

    # Calculate centroids
    centroids = (
        pca_df
        .groupby("sample")[["PC1", "PC2"]]
        .mean()
    )

    if reference not in centroids.index:
        raise ValueError(
            f"Reference '{reference}' not found in samples."
        )

    ref = centroids.loc[reference]

    # Euclidean distance from reference
    distances = np.sqrt(
        (centroids["PC1"] - ref["PC1"])**2 +
        (centroids["PC2"] - ref["PC2"])**2
    )

    distances.name = f"distance_from_{reference}"

    return distances, centroids

def plot_pca_centroid_distances(
    distances,
    reference="WT",
    figsize=(6, 3.5),
    dpi=300,
    exclude_reference=True,
    show_values=True,
    show=True
):
    """
    Plot distances of PCA centroids from a reference group.

    Parameters
    ----------
    distances : pd.Series or dict
        Distance of each sample centroid from the reference centroid.

    reference : str
        Name of the reference group.

    figsize : tuple
        Figure size.

    dpi : int
        Figure resolution.

    exclude_reference : bool
        If True, remove the reference group (distance = 0).

    show_values : bool
        Display numerical distance next to each point.

    show : bool
        Whether to display the figure.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    # Convert to Series
    distances = pd.Series(distances, dtype=float)

    # Remove reference
    if exclude_reference:
        distances = distances.drop(reference, errors="ignore")

    # Sort by distance
    distances = distances.sort_values()

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    y_positions = np.arange(len(distances))

    # Vertical reference line
    ax.axvline(
        0,
        linewidth=1.2,
        zorder=1
    )

    # Horizontal branches
    ax.hlines(
        y=y_positions,
        xmin=0,
        xmax=distances.values,
        linewidth=1.5,
        zorder=1
    )

    # Points
    ax.scatter(
        distances.values,
        y_positions,
        s=55,
        edgecolor="black",
        linewidth=0.6,
        zorder=3
    )

    # Sample names on y-axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        distances.index,
        fontsize=10
    )

    # Numerical values
    if show_values:

        offset = distances.max() * 0.025

        for y, value in zip(
            y_positions,
            distances.values
        ):
            ax.text(
                value + offset,
                y,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=9
            )

    # Axis label
    ax.set_xlabel(
        f"Centroid distance from {reference}",
        fontsize=11
    )

    ax.tick_params(
        axis="x",
        labelsize=9,
        direction="out"
    )

    ax.tick_params(
        axis="y",
        length=0
    )

    # Clean publication style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.grid(
        axis="x",
        linewidth=0.5,
        alpha=0.2
    )

    # Space for labels
    ax.set_xlim(
        0,
        distances.max() * 1.15
    )

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax

def _compare_pca_to_wt(
    pca_df,
    wt_label="WT",
    test="mannwhitney"
):
    results = []

    wt_pc1 = pca_df.loc[
        pca_df["sample"] == wt_label,
        "PC1"
    ]

    wt_pc2 = pca_df.loc[
        pca_df["sample"] == wt_label,
        "PC2"
    ]

    mutants = [
        x for x in pca_df["sample"].unique()
        if x != wt_label
    ]

    for mutant in mutants:

        mut_pc1 = pca_df.loc[
            pca_df["sample"] == mutant,
            "PC1"
        ]

        mut_pc2 = pca_df.loc[
            pca_df["sample"] == mutant,
            "PC2"
        ]

        stat1, p1 = mannwhitneyu(
            wt_pc1,
            mut_pc1,
            alternative="two-sided"
        )

        stat2, p2 = mannwhitneyu(
            wt_pc2,
            mut_pc2,
            alternative="two-sided"
        )

        results.append({
            "sample": mutant,

            "WT_PC1_mean": wt_pc1.mean(),
            "Mut_PC1_mean": mut_pc1.mean(),
            "PC1_p": p1,

            "WT_PC2_mean": wt_pc2.mean(),
            "Mut_PC2_mean": mut_pc2.mean(),
            "PC2_p": p2,
        })

    results = pd.DataFrame(results)
    #results = results[['sample', 'PC1_p', 'PC2_p']]
    
    return results

def beautiful_pca_plot(
    pca_df,
    centroids,
    explained_variance=None,
    figsize=(5, 4.5),
    dpi=300,
    point_size=30,
    centroid_size=90,
    alpha=0.6,
    show=True,
):
    """
    Publication-quality PCA scatter plot with group centroids.

    Parameters
    ----------
    pca_df : pd.DataFrame
        DataFrame containing:
        'PC1', 'PC2', and 'sample'.

    centroids : pd.DataFrame
        DataFrame indexed by sample containing:
        'PC1' and 'PC2'.

    explained_variance : array-like, optional
        Explained variance ratios from PCA.

    figsize : tuple
        Figure size.

    dpi : int
        Figure resolution.

    point_size : float
        Size of individual observations.

    centroid_size : float
        Size of centroid markers.

    alpha : float
        Transparency of individual observations.

    show : bool
        Whether to display the plot.

    Returns
    -------
    fig, ax
    """

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    samples = pca_df["sample"].unique()

    # Use matplotlib's default color cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    color_map = {
        sample: colors[i % len(colors)]
        for i, sample in enumerate(samples)
    }

    # --------------------------------------------------
    # Individual observations
    # --------------------------------------------------

    for sample in samples:

        subset = pca_df[
            pca_df["sample"] == sample
        ]

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=point_size,
            alpha=alpha,
            color=color_map[sample],
            edgecolor="white",
            linewidth=0.3,
            label=sample,
            zorder=2
        )

    # --------------------------------------------------
    # Centroids
    # --------------------------------------------------

    for sample in centroids.index:

        x = centroids.loc[sample, "PC1"]
        y = centroids.loc[sample, "PC2"]

        ax.scatter(
            x,
            y,
            marker="X",
            s=centroid_size,
            color=color_map.get(sample),
            edgecolor="black",
            linewidth=0.7,
            zorder=5
        )

    # --------------------------------------------------
    # Zero reference lines
    # --------------------------------------------------

    ax.axhline(
        0,
        linewidth=0.7,
        linestyle="--",
        alpha=0.3,
        zorder=0
    )

    ax.axvline(
        0,
        linewidth=0.7,
        linestyle="--",
        alpha=0.3,
        zorder=0
    )

    # --------------------------------------------------
    # Axis labels
    # --------------------------------------------------

    if explained_variance is not None:

        ax.set_xlabel(
            f"PC1 ({explained_variance[0] * 100:.1f}%)",
            fontsize=11
        )

        ax.set_ylabel(
            f"PC2 ({explained_variance[1] * 100:.1f}%)",
            fontsize=11
        )

    else:

        ax.set_xlabel(
            "PC1",
            fontsize=11
        )

        ax.set_ylabel(
            "PC2",
            fontsize=11
        )

    # --------------------------------------------------
    # Ticks
    # --------------------------------------------------

    ax.tick_params(
        axis="both",
        labelsize=9,
        direction="out",
        length=4,
        width=0.8
    )

    # --------------------------------------------------
    # Spines
    # --------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------

    ax.legend(
        frameon=False,
        fontsize=9,
        markerscale=1.1,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax

def pairwise_permanova(
    dfs,
    data,
    columns,
    permutations=9999,
    correction="fdr_bh"
):
    """
    Pairwise PERMANOVA between all samples.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        DataFrames in the same order as `data`.

    data : list[dict]
        Sample information containing at least:
        {"name": sample_name, "path": path}

    columns : list[str]
        Variables used for the analysis.

    permutations : int
        Number of permutations.

    correction : str
        Multiple-testing correction method.
        Examples: "fdr_bh", "holm", "bonferroni".

    Returns
    -------
    results : pd.DataFrame
        Pairwise PERMANOVA results.
    """

    # --------------------------------
    # Combine all samples
    # --------------------------------

    combined = []

    for df, sample in zip(dfs, data):

        tmp = df[columns].copy()
        tmp["sample"] = sample["name"]

        combined.append(tmp)

    df_all = pd.concat(
        combined,
        ignore_index=True
    )

    df_all = (
        df_all
        .dropna(subset=columns)
        .reset_index(drop=True)
    )

    sample_names = [
        sample["name"]
        for sample in data
    ]

    results = []

    # --------------------------------
    # All pairwise comparisons
    # --------------------------------

    for group1, group2 in combinations(sample_names, 2):

        subset = df_all[
            df_all["sample"].isin([group1, group2])
        ].copy()

        # Standardize variables
        X = StandardScaler().fit_transform(
            subset[columns]
        )

        # Euclidean distance matrix
        distance_matrix = squareform(
            pdist(X, metric="euclidean")
        )

        dm = DistanceMatrix(distance_matrix)

        # PERMANOVA
        result = permanova(
            dm,
            grouping=subset["sample"].to_numpy(),
            permutations=permutations
        )

        results.append({
            "group1": group1,
            "group2": group2,
            "n1": (subset["sample"] == group1).sum(),
            "n2": (subset["sample"] == group2).sum(),
            "pseudo_F": result["test statistic"],
            "p_value": result["p-value"]
        })

    results = pd.DataFrame(results)

    # --------------------------------
    # Multiple-testing correction
    # --------------------------------

    reject, p_adj, _, _ = multipletests(
        results["p_value"],
        method=correction
    )

    results["p_adj"] = p_adj
    results["significant"] = reject

    return results