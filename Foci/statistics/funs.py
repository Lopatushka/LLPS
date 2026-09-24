# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

def beautiful_pca_plot(
    pca_df,
    centroids,
    explained_variance=None,
    figsize=(8, 6),
    dpi=300,
):
    
    #colors = {
        #"WT": "#4C72B0",
        #"MGS1": "#55A868",
        #"MGS2": "#C44E52",
        #"MGS3": "#8172B2",
        #"MGS4": "#CCB974",
        #"MGS5": "#64B5CD",
    #}

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    # -------------------------
    # Cells
    # -------------------------

    for sample in pca_df["sample"].unique():

        subset = pca_df[
            pca_df["sample"] == sample
        ]

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=30,
            alpha=0.35,
            #color=colors.get(sample),
            label=sample,
            edgecolors="none"
        )

    # -------------------------
    # Centroids
    # -------------------------

    for sample in centroids.index:

        x = centroids.loc[sample, "PC1"]
        y = centroids.loc[sample, "PC2"]

        ax.scatter(
            x,
            y,
            marker="X",
            s=100,
            #color=colors.get(sample),
            edgecolor="black",
            linewidth=0.5,
            zorder=10,
        )

        ax.annotate(
            sample,
            (x, y),
            xytext=(-10, 5),
            textcoords="offset points",
            fontsize=6,
            weight="bold"
        )

    # -------------------------
    # Labels
    # -------------------------

    if explained_variance is not None:

        ax.set_xlabel(
            f"PC1 ({explained_variance[0]*100:.1f}%)",
            fontsize=12
        )

        ax.set_ylabel(
            f"PC2 ({explained_variance[1]*100:.1f}%)",
            fontsize=12
        )

    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    # -------------------------
    # Style
    # -------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        alpha=0.2,
        linestyle="--"
    )

    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    return fig, ax

def compare_pca_to_wt(
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