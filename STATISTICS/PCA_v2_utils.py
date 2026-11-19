"""PCA of per-cell foci metrics: preparation, fit, plots and replicate-aware tests.

Used by PCA_v2.ipynb. Input: the aggregated_data.csv of each condition and
replicate (one row per cell, written by agg.py). Loading, design checks,
display names and plot style are shared with METRICS_utils.py.

PERMANOVA / PERMDISP are implemented here directly for Euclidean distances
(sums of squares from centroids, no distance matrix), so they are fast for
thousands of cells and can be run at the replicate level with exact
permutations.
"""

from itertools import combinations
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
from statsmodels.stats.multitest import multipletests

from METRICS_utils import (  # noqa: F401  (re-exported for the notebook)
    Path, load_conditions, design_table, check_design, apply_min_foci,
    missing_table, compare_metrics, comparison_pairs, plot_superplots,
    stars, pretty, DISPLAY_NAMES, STYLE, STAR_STYLE, NS_STYLE,
)


def condition_palette(conditions):
    """Colour per condition, shared by every plot of the notebook."""
    return dict(zip(conditions, sns.color_palette("colorblind", len(conditions))))


# =====================================================
# 1. FEATURE PREPARATION
# =====================================================

def prepare_features(df, features, log_features=(), missing="drop"):
    """Transform, handle missing values and z-score the features.

    log_features : features replaced by log10(1 + x) before scaling, which
                   reduces the weight of a few extreme cells. Only for
                   features >= 0.
    missing      : "drop"   - cells with any missing feature are removed;
                   "median" - missing values are filled with the median of
                              the feature over all cells (keeps cells, but
                              pulls them towards the centre).

    Returns (data, Z, info): `data` = kept cells with the original columns,
    `Z` = DataFrame of z-scored features (same index), `info` = cells kept per
    condition.
    """
    features = [f for f in features if f in df.columns]
    data = df.copy()
    X = data[features].astype(float).copy()
    for f in log_features:
        if f in X.columns:
            if (X[f].dropna() < 0).any():
                raise ValueError(f"{f} has negative values: it can't be log-transformed")
            X[f] = np.log10(1 + X[f])

    n_before = data.groupby("condition", observed=True).size()
    if missing == "drop":
        keep = X.notna().all(axis=1)
        data, X = data[keep], X[keep]
    elif missing == "median":
        X = X.fillna(X.median())
    else:
        raise ValueError('missing must be "drop" or "median"')

    sd = X.std(ddof=0)
    constant = sd[sd == 0].index.tolist()
    if constant:
        print(f"Removed constant features (no information): {', '.join(constant)}")
        X = X.drop(columns=constant)
    Z = (X - X.mean()) / X.std(ddof=0)

    n_after = data.groupby("condition", observed=True).size()
    info = pd.DataFrame({"cells": n_before, "kept": n_after.reindex(n_before.index, fill_value=0)})
    info["kept_%"] = (100 * info["kept"] / info["cells"]).round(0)
    return data, Z, info


def plot_feature_distributions(Z, names=None, ncols=6):
    """Histogram of each z-scored feature (after the log transform)."""
    cols = list(Z.columns)
    nrows = int(np.ceil(len(cols) / ncols))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 1.9 * nrows), squeeze=False)
        for ax, f in zip(axes.flat, cols):
            ax.hist(Z[f], bins=30, color="#7a9cc6", edgecolor="white", linewidth=0.4)
            ax.set_title(pretty(f, names), fontsize=9)
            ax.set_yticks([])
            sns.despine(ax=ax, left=True)
            ax.grid(False)
        for ax in list(axes.flat)[len(cols):]:
            ax.axis("off")
        fig.supxlabel("z-score", fontsize=9)
        fig.tight_layout()
    return fig


# =====================================================
# 2. CORRELATION
# =====================================================

def correlation_matrix(Z, method="spearman"):
    """Feature correlation, rows/columns ordered by hierarchical clustering."""
    corr = Z.corr(method=method)
    if len(corr) > 2:
        dist = 1 - corr.abs().to_numpy()
        np.fill_diagonal(dist, 0)
        order = hierarchy.leaves_list(hierarchy.linkage(squareform(dist, checks=False), "average"))
        corr = corr.iloc[order, order]
    return corr


def redundant_pairs(corr, threshold=0.85):
    """Feature pairs with |rho| >= threshold: candidates to drop one of them."""
    rows = []
    cols = list(corr.columns)
    for i, j in combinations(range(len(cols)), 2):
        r = corr.iloc[i, j]
        if abs(r) >= threshold:
            rows.append(dict(feature_1=cols[i], feature_2=cols[j], rho=round(r, 3)))
    return pd.DataFrame(rows, columns=["feature_1", "feature_2", "rho"])


def plot_correlation(corr, names=None, title="Spearman correlation between features"):
    """Lower-triangle heatmap of the correlation matrix."""
    labels = [pretty(c, names) for c in corr.columns]
    n = len(labels)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(0.55 * n + 3, 0.5 * n + 2))
        sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True,
                    annot=n <= 16, fmt=".2f", annot_kws={"fontsize": 7},
                    linewidths=1, linecolor="white",
                    cbar_kws={"label": "Spearman ρ", "shrink": 0.6},
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.tick_params(length=0)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        ax.set_title(title, pad=10)
        fig.tight_layout()
    return fig


# =====================================================
# 3. PCA FIT
# =====================================================

def fit_pca(Z, data):
    """PCA with all components; returns (pca, scores) where `scores` is
    `data` with columns PC1..PCk added."""
    pca = PCA().fit(Z.to_numpy())
    pcs = pca.transform(Z.to_numpy())
    scores = data.copy()
    for k in range(pcs.shape[1]):
        scores[f"PC{k + 1}"] = pcs[:, k]
    return pca, scores


def parallel_analysis(Z, n_iter=200, quantile=95, seed=0):
    """Explained variance (%) expected by chance, per component.

    Each feature is shuffled independently (destroying the correlations
    between features) and the PCA is refitted. A component is meaningful if
    it explains more variance than the `quantile` of these random PCAs.
    """
    rng = np.random.default_rng(seed)
    X = Z.to_numpy()
    out = np.empty((n_iter, X.shape[1]))
    for i in range(n_iter):
        shuffled = np.column_stack([rng.permutation(col) for col in X.T])
        out[i] = PCA().fit(shuffled).explained_variance_ratio_ * 100
    return np.percentile(out, quantile, axis=0)


def plot_scree(pca, chance=None, max_pcs=12):
    """Explained variance per PC (bars), cumulative (line), chance level (dashes)."""
    var = pca.explained_variance_ratio_[:max_pcs] * 100
    k = np.arange(1, len(var) + 1)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(1.0 + 0.5 * len(var), 3.6))
        meaningful = var > chance[:len(var)] if chance is not None else np.ones(len(var), bool)
        ax.bar(k, var, color=np.where(meaningful, "#3b75af", "#c9d6e6"), width=0.7, zorder=2)
        for x, v, m in zip(k, var, meaningful):
            inside = v >= 8
            ax.text(x, v - 1.2 if inside else v + 0.8, f"{v:.0f}", ha="center",
                    va="top" if inside else "bottom", fontsize=8,
                    color=("white" if m else "#333333") if inside else "#333333", zorder=4)
        ax.plot(k, np.cumsum(var), "-o", color="#333333", markersize=4, linewidth=1.3,
                label="cumulative", zorder=3)
        if chance is not None:
            ax.plot(k, chance[:len(var)], "--", color="#d1495b", linewidth=1.3,
                    label="chance level (parallel analysis)", zorder=3)
        ax.set_xticks(k)
        ax.set_xticklabels([f"PC{i}" for i in k])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("explained variance (%)")
        ax.set_ylim(0, 105)
        ax.set_title("How much of the variation each component captures", pad=10)
        ax.legend(frameon=False, loc="center right")
        sns.despine(ax=ax)
        fig.tight_layout()
    return fig


def loadings_table(pca, features, n_pcs=3):
    """Correlation of each feature with each PC (loading x sqrt(eigenvalue)).

    For z-scored features this is the Pearson correlation between the
    feature and the PC scores: -1..1, comparable between PCs.
    """
    n_pcs = min(n_pcs, pca.n_components_)
    load = pca.components_[:n_pcs].T * np.sqrt(pca.explained_variance_[:n_pcs])
    return pd.DataFrame(load, index=features, columns=[f"PC{i + 1}" for i in range(n_pcs)])


def plot_loadings(loadings, pca, names=None):
    """Heatmap of feature-PC correlations, features sorted by PC1."""
    L = loadings.sort_values(loadings.columns[0])
    var = pca.explained_variance_ratio_ * 100
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(2.8 + 1.1 * L.shape[1], 0.42 * len(L) + 1.4))
        sns.heatmap(L, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt="+.2f",
                    annot_kws={"fontsize": 8}, linewidths=1.5, linecolor="white",
                    cbar_kws={"label": "correlation with PC", "shrink": 0.7},
                    yticklabels=[pretty(f, names) for f in L.index],
                    xticklabels=[f"{c}\n{var[i]:.0f}%" for i, c in enumerate(L.columns)], ax=ax)
        ax.tick_params(length=0)
        ax.set_title("What each component measures", pad=10)
        fig.tight_layout()
    return fig


def plot_correlation_circle(loadings, pca, pcs=("PC1", "PC2"), names=None, min_length=0.3):
    """Arrows = correlation of each feature with two PCs.

    Long arrows are well represented by these two PCs; arrows pointing the
    same way are positively correlated, opposite ways negatively, at right
    angles uncorrelated. Features with arrows shorter than `min_length` are
    drawn faint and unlabelled.
    """
    a, b = pcs
    var = pca.explained_variance_ratio_ * 100
    ia, ib = int(a[2:]) - 1, int(b[2:]) - 1
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        t = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(t), np.sin(t), color="#bbbbbb", linewidth=1)
        ax.plot(0.5 * np.cos(t), 0.5 * np.sin(t), color="#e3e3e3", linewidth=0.8, linestyle="--")
        ax.axhline(0, color="#dddddd", linewidth=0.8)
        ax.axvline(0, color="#dddddd", linewidth=0.8)
        labels = []
        for f, (x, y) in loadings[[a, b]].iterrows():
            length = np.hypot(x, y)
            strong = length >= min_length
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#2b5d8a" if strong else "#c4cfdb",
                                        linewidth=1.6 if strong else 1, mutation_scale=12))
            if strong:
                ux, uy = x / length, y / length
                labels.append([f, x, y, x + 0.09 * ux, y + 0.09 * uy])
        # Spread labels on each side so that they don't overlap (min vertical gap)
        gap = 0.075
        for side in (1, -1):
            group = sorted([l for l in labels if np.sign(l[3] or 1) == side], key=lambda l: l[4])
            for _ in range(50):
                moved = False
                for k in range(1, len(group)):
                    d = group[k][4] - group[k - 1][4]
                    if d < gap:
                        shift = (gap - d) / 2
                        group[k][4] += shift
                        group[k - 1][4] -= shift
                        moved = True
                if not moved:
                    break
        for f, x, y, lx, ly in labels:
            ux, uy = np.array([x, y]) / np.hypot(x, y)
            if np.hypot(lx - (x + 0.09 * ux), ly - (y + 0.09 * uy)) > 0.03:
                ax.plot([x, lx], [y, ly], color="#bbbbbb", linewidth=0.6, zorder=0)
            ax.text(lx, ly, pretty(f, names), fontsize=8.5, color="#1a1a1a",
                    ha="left" if lx >= 0 else "right", va="center")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{a} ({var[ia]:.1f}%)")
        ax.set_ylabel(f"{b} ({var[ib]:.1f}%)")
        ax.set_title("Correlation circle", pad=10)
        sns.despine(ax=ax)
        fig.tight_layout()
    return fig


# =====================================================
# 4. CENTROIDS AND PCA SCATTER
# =====================================================

def _resample(arrays, rng):
    """Hierarchical resample: replicates with replacement, then cells."""
    picks = rng.integers(len(arrays), size=len(arrays))
    return np.concatenate([arrays[i][rng.integers(len(arrays[i]), size=len(arrays[i]))]
                           for i in picks])


def bootstrap_centroids(scores, columns, n_boot=2000, seed=0):
    """Bootstrap distribution of each condition's centroid in `columns`.

    Resamples replicates, then cells within replicates, so the spread
    includes the variability between experiments (cells only if there is
    one replicate). Returns {condition: (centroid, boot array n_boot x d)}.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for cond, g in scores.groupby("condition", observed=True):
        arrays = [r[columns].to_numpy(float) for _, r in g.groupby("replicate")]
        boot = np.array([_resample(arrays, rng).mean(axis=0) for _ in range(n_boot)])
        out[cond] = (g[columns].to_numpy(float).mean(axis=0), boot)
    return out


def _ellipse(ax, center, cov, confidence, color, **kw):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = np.maximum(vals[order], 0), vecs[:, order]
    w, h = 2 * np.sqrt(vals * stats.chi2.ppf(confidence, 2))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse(center, w, h, angle=angle, facecolor=color, alpha=0.22,
                         edgecolor="none", zorder=3))
    ax.add_patch(Ellipse(center, w, h, angle=angle, facecolor="none", edgecolor=color,
                         linewidth=1.8, zorder=4, **kw))


def plot_pca_scatter(scores, pca, pcs=("PC1", "PC2"), palette=None, confidence=0.95,
                     n_boot=2000, show_replicates=True, title="Cells in PCA space"):
    """All cells, condition centroids with confidence ellipses, replicate centroids.

    small dots : cells
    ellipse    : confidence region of the condition centroid (hierarchical
                 bootstrap: includes the variability between replicates)
    big dot    : condition centroid
    diamonds   : replicate centroids (if >1 replicate)
    """
    a, b = pcs
    conditions = list(scores["condition"].cat.categories)
    palette = palette or condition_palette(conditions)
    var = pca.explained_variance_ratio_ * 100
    boots = bootstrap_centroids(scores, [a, b], n_boot=n_boot)
    multi_rep = scores.groupby("condition", observed=True)["replicate"].nunique().max() > 1

    with plt.rc_context({**STYLE, "axes.grid.axis": "both"}):
        fig, ax = plt.subplots(figsize=(7.4, 6.2))
        ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
        ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=0)
        for cond in conditions:
            g = scores[scores["condition"] == cond]
            if g.empty:
                continue
            c = palette[cond]
            ax.scatter(g[a], g[b], s=14, color=c, alpha=0.35, linewidths=0, zorder=1)
            centre, boot = boots[cond]
            if len(boot) > 2 and np.isfinite(boot).all():
                _ellipse(ax, centre, np.cov(boot, rowvar=False), confidence, c)
            if multi_rep and show_replicates:
                rc = g.groupby("replicate")[[a, b]].mean()
                ax.scatter(rc[a], rc[b], marker="D", s=42, color=c, edgecolor="black",
                           linewidths=0.8, zorder=5)
            ax.scatter(*centre, s=120, color=c, edgecolor="black", linewidths=1.2, zorder=6,
                       label=f"{cond}  (n = {len(g)})")
        ax.set_xlabel(f"{a} ({var[int(a[2:]) - 1]:.1f}%)")
        ax.set_ylabel(f"{b} ({var[int(b[2:]) - 1]:.1f}%)")
        ax.set_title(title, pad=10)
        handles, labels = ax.get_legend_handles_labels()
        if multi_rep and show_replicates:
            handles.append(plt.Line2D([], [], marker="D", linestyle="", color="#999999",
                                      markeredgecolor="black", label="replicate centroid"))
        handles.append(plt.Line2D([], [], color="#777777", linewidth=1.8,
                                  label=f"{confidence:.0%} CI of centroid"))
        ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
        sns.despine(ax=ax)
        fig.tight_layout()
    return fig


def plot_pca_facets(scores, pca, pcs=("PC1", "PC2"), palette=None, ncols=3):
    """One panel per condition: its cells in colour, all other cells in grey."""
    a, b = pcs
    conditions = list(scores["condition"].cat.categories)
    palette = palette or condition_palette(conditions)
    var = pca.explained_variance_ratio_ * 100
    ncols = min(ncols, len(conditions))
    nrows = int(np.ceil(len(conditions) / ncols))
    with plt.rc_context({**STYLE, "axes.grid.axis": "both"}):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, cond in zip(axes.flat, conditions):
            other = scores[scores["condition"] != cond]
            own = scores[scores["condition"] == cond]
            ax.scatter(other[a], other[b], s=8, color="#d9d9d9", linewidths=0, zorder=1)
            ax.scatter(own[a], own[b], s=12, color=palette[cond], alpha=0.75, linewidths=0, zorder=2)
            ax.scatter(own[a].mean(), own[b].mean(), s=90, color=palette[cond], edgecolor="black",
                       linewidths=1.1, zorder=3)
            ax.set_title(f"{cond}  (n = {len(own)})")
            sns.despine(ax=ax)
        for ax in list(axes.flat)[len(conditions):]:
            ax.axis("off")
        fig.supxlabel(f"{a} ({var[int(a[2:]) - 1]:.1f}%)", fontsize=10)
        fig.supylabel(f"{b} ({var[int(b[2:]) - 1]:.1f}%)", fontsize=10)
        fig.tight_layout()
    return fig


def plot_pca_colored(scores, pca, column, pcs=("PC1", "PC2"), names=None, log=True,
                     cmap="viridis"):
    """Cells in PCA space coloured by a continuous column (e.g. expression)."""
    a, b = pcs
    var = pca.explained_variance_ratio_ * 100
    v = scores[column].astype(float)
    shown = np.log10(v.clip(lower=v[v > 0].min() if (v > 0).any() else 1)) if log else v
    with plt.rc_context({**STYLE, "axes.grid.axis": "both"}):
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        order = np.argsort(shown.to_numpy())
        sc = ax.scatter(scores[a].to_numpy()[order], scores[b].to_numpy()[order],
                        c=shown.to_numpy()[order], cmap=cmap, s=16, linewidths=0, alpha=0.85)
        cb = fig.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label(("log10 " if log else "") + pretty(column, names))
        cb.outline.set_visible(False)
        ax.set_xlabel(f"{a} ({var[int(a[2:]) - 1]:.1f}%)")
        ax.set_ylabel(f"{b} ({var[int(b[2:]) - 1]:.1f}%)")
        ax.set_title(f"PCA coloured by {pretty(column, names).lower()}", pad=10)
        sns.despine(ax=ax)
        fig.tight_layout()
    return fig


# =====================================================
# 5. MULTIVARIATE TESTS
# =====================================================

def _pseudo_f(X, codes, k):
    """PERMANOVA pseudo-F and R2 for Euclidean distances, from sums of squares."""
    n = len(X)
    sst = ((X - X.mean(axis=0)) ** 2).sum()
    counts = np.bincount(codes, minlength=k)
    sums = np.zeros((k, X.shape[1]))
    np.add.at(sums, codes, X)
    ssa = (sums ** 2 / counts[:, None]).sum() - n * (X.mean(axis=0) ** 2).sum()
    ssw = sst - ssa
    return (ssa / (k - 1)) / (ssw / (n - k)), ssa / sst


def _labelings(codes, n_perm, rng):
    """All distinct relabelings if there are at most n_perm (exact test),
    else n_perm random ones. Returns (list of label arrays, exact)."""
    counts = np.bincount(codes)
    n_total = math.factorial(len(codes))
    for c in counts:
        n_total //= math.factorial(c)
    if n_total > n_perm:
        return [rng.permutation(codes) for _ in range(n_perm)], False

    def assign(free, group):
        if group == len(counts) - 1:
            yield {group: free}
            return
        for chosen in combinations(free, counts[group]):
            rest = tuple(i for i in free if i not in chosen)
            for tail in assign(rest, group + 1):
                yield {group: chosen, **tail}

    out = []
    for a in assign(tuple(range(len(codes))), 0):
        lab = np.empty(len(codes), int)
        for g, idx in a.items():
            lab[list(idx)] = g
        out.append(lab)
    return out, True


def permanova(X, groups, n_perm=9999, seed=0):
    """PERMANOVA (Euclidean). Returns dict(F, R2, p, n, groups, exact).

    R2 = fraction of the total variance explained by the grouping. With few
    observations (replicate level) all distinct labelings are enumerated, so
    p is exact; the smallest possible p is then 1 / number of labelings.
    """
    X = np.asarray(X, float)
    cats, codes = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    k = len(cats)
    if k < 2 or len(X) <= k:
        return dict(F=np.nan, R2=np.nan, p=np.nan, n=len(X), groups=k, exact=False, p_min=np.nan)
    F, R2 = _pseudo_f(X, codes, k)
    rng = np.random.default_rng(seed)
    perms, exact = _labelings(codes, n_perm, rng)
    Fp = np.array([_pseudo_f(X, p, k)[0] for p in perms])
    if exact:
        p = np.mean(Fp >= F - 1e-12)
        # smallest p this design can give: share of labelings reaching the top F
        p_min = np.mean(Fp >= Fp.max() * (1 - 1e-9))
    else:
        p = (1 + np.sum(Fp >= F - 1e-12)) / (1 + len(perms))
        p_min = 1 / (1 + len(perms))
    return dict(F=F, R2=R2, p=p, n=len(X), groups=k, exact=exact, p_min=p_min)


def permdisp(X, groups, n_perm=9999, seed=0):
    """PERMDISP: do the groups differ in spread (distance of cells to their
    group centroid)? A significant PERMDISP means a PERMANOVA difference may
    come from different spread, not (only) different location."""
    X = np.asarray(X, float)
    groups = np.asarray(groups).astype(str)
    cats = np.unique(groups)
    d = np.empty(len(X))
    for c in cats:
        m = groups == c
        d[m] = np.linalg.norm(X[m] - X[m].mean(axis=0), axis=1)
    F = stats.f_oneway(*[d[groups == c] for c in cats]).statistic
    rng = np.random.default_rng(seed)
    Fp = np.array([stats.f_oneway(*[d[p == c] for c in cats]).statistic
                   for p in (rng.permutation(groups) for _ in range(n_perm))])
    return dict(F=F, p=(1 + np.sum(Fp >= F)) / (1 + n_perm),
                mean_distance={c: float(d[groups == c].mean()) for c in cats})


def multivariate_tests(scores, Z, pairs, n_perm=9999, seed=0):
    """PERMANOVA / PERMDISP on all features (z-scored), overall and per pair.

    level "cells"      : every cell is an observation (descriptive: cells of
                         one replicate are not independent).
    level "replicates" : replicate centroids are the observations (n =
                         replicates); exact permutation p-values.
    q-values: Benjamini-Hochberg over the pairwise tests of each level.
    """
    X = Z.loc[scores.index].to_numpy(float)
    cond = scores["condition"].astype(str).to_numpy()
    rep_key = scores["condition"].astype(str) + "|" + scores["replicate"].astype(str)
    R = pd.DataFrame(X, index=scores.index).groupby(rep_key.to_numpy()).mean()
    r_cond = np.array([k.split("|")[0] for k in R.index])
    multi_rep = pd.Series(r_cond).value_counts().min() >= 2

    rows = []
    comparisons = [("all conditions", None)] + [(f"{a} vs {b}", (a, b)) for a, b in pairs]
    for label, pair in comparisons:
        m = np.ones(len(cond), bool) if pair is None else np.isin(cond, pair)
        pm = permanova(X[m], cond[m], n_perm=n_perm, seed=seed)
        pd_ = permdisp(X[m], cond[m], n_perm=min(n_perm, 1999), seed=seed)
        rows.append(dict(comparison=label, level="cells", n=pm["n"], F=pm["F"], R2=pm["R2"],
                         p=pm["p"], p_permdisp=pd_["p"]))
        if multi_rep:
            mr = np.ones(len(r_cond), bool) if pair is None else np.isin(r_cond, pair)
            pr = permanova(R.to_numpy()[mr], r_cond[mr], n_perm=n_perm, seed=seed)
            rows.append(dict(comparison=label, level="replicates", n=pr["n"], F=pr["F"],
                             R2=pr["R2"], p=pr["p"], p_permdisp=np.nan, p_min=pr["p_min"]))
    res = pd.DataFrame(rows)
    res["q"] = np.nan
    for level in res["level"].unique():
        m = (res["level"] == level) & (res["comparison"] != "all conditions") & res["p"].notna()
        if m.any():
            res.loc[m, "q"] = multipletests(res.loc[m, "p"], method="fdr_bh")[1]
    if "p_min" not in res:
        res["p_min"] = np.nan
    return res[["comparison", "level", "n", "F", "R2", "p", "q", "p_min", "p_permdisp"]]


def centroid_distances(scores, Z, reference, n_boot=2000, confidence=0.95, seed=0):
    """Distance of each condition's centroid from the reference centroid in the
    full z-scored feature space, with a hierarchical-bootstrap CI.

    Unit: standard deviations (root of the summed squared standardised
    differences over all features). Uses all features, not only PC1-PC2.
    """
    cols = list(Z.columns)
    S = pd.concat([scores[["condition", "replicate"]], Z.loc[scores.index]], axis=1)
    boots = bootstrap_centroids(S, cols, n_boot=n_boot, seed=seed)
    ref_c, ref_b = boots[reference]
    half = (1 - confidence) / 2
    rows = []
    for cond, (c, b) in boots.items():
        if cond == reference:
            continue
        d_boot = np.linalg.norm(b - ref_b, axis=1)
        diff = c - ref_c
        top = np.argsort(-np.abs(diff))[:3]
        rows.append(dict(condition=cond, distance=float(np.linalg.norm(diff)),
                         ci_low=float(np.quantile(d_boot, half)),
                         ci_high=float(np.quantile(d_boot, 1 - half)),
                         main_features=", ".join(f"{cols[i]} ({diff[i]:+.2f})" for i in top)))
    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)


def plot_centroid_distances(dist, reference, palette, mv=None):
    """Distance from the reference centroid with CI; stars from PERMANOVA q."""
    with plt.rc_context({**STYLE, "axes.grid.axis": "x"}):
        fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(dist) + 1.5))
        y = np.arange(len(dist))[::-1]
        for yi, (_, r) in zip(y, dist.iterrows()):
            c = palette[r["condition"]]
            ax.plot([r["ci_low"], r["ci_high"]], [yi, yi], color=c, linewidth=3,
                    solid_capstyle="round", alpha=0.6, zorder=2)
            ax.scatter(r["distance"], yi, s=110, color=c, edgecolor="black", linewidths=1.1, zorder=3)
            if mv is not None:
                sel = mv[(mv["comparison"] == f"{r['condition']} vs {reference}")]
                level = "replicates" if (sel["level"] == "replicates").any() else "cells"
                q = sel.loc[sel["level"] == level, "q"]
                s = stars(q.iloc[0]) if len(q) else ""
                if s:
                    ax.text(r["ci_high"], yi, f"  {s}", va="center", ha="left",
                            **(NS_STYLE if s == "ns" else {**STAR_STYLE, "fontsize": 13}))
        ax.set_yticks(y)
        ax.set_yticklabels(dist["condition"])
        ax.set_ylim(-0.7, len(dist) - 0.3)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(left=0)
        ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
        ax.set_xlabel(f"distance from {reference} centroid (SD units, all features)")
        ax.set_title(f"How different is each condition from {reference}?", pad=10)
        sns.despine(ax=ax, left=True)
        fig.tight_layout()
    return fig


# =====================================================
# 6. OUTLIERS
# =====================================================

def find_outliers(scores, Z, confidence=0.999):
    """Multivariate outliers, judged within each condition.

    Mahalanobis distance of each cell from its own condition's centroid, in
    the full z-scored feature space, against the chi-square cut-off.
    Within-condition, because a pooled detection would flag a whole condition
    whose phenotype really differs. Classical covariance, because robust
    estimators (MCD) are unstable with few cells per feature and with
    features that pile up at one value (e.g. nn_gap = 0). Returns a DataFrame
    with `mahalanobis` and `outlier` (index of Z); conditions with too few
    cells (<= 2 x number of features) are skipped.
    """
    p = Z.shape[1]
    cut = stats.chi2.ppf(confidence, p)
    out = pd.DataFrame({"mahalanobis": np.nan, "outlier": False}, index=Z.index)
    for cond, g in scores.groupby("condition", observed=True):
        X = Z.loc[g.index].to_numpy()
        if len(X) <= 2 * p:
            print(f"{cond}: only {len(X)} cells for {p} features, outlier detection skipped")
            continue
        d2 = EmpiricalCovariance().fit(X).mahalanobis(X)
        out.loc[g.index, "mahalanobis"] = np.sqrt(d2)
        out.loc[g.index, "outlier"] = d2 > cut
    return out


def outlier_order(flags):
    """Index of the flagged cells, most extreme first (numbering of the plot)."""
    return flags[flags["outlier"]].sort_values("mahalanobis", ascending=False).index


def plot_outliers(scores, pca, flags, pcs=("PC1", "PC2"), palette=None):
    """PCA scatter with the flagged cells circled in red."""
    a, b = pcs
    conditions = list(scores["condition"].cat.categories)
    palette = palette or condition_palette(conditions)
    var = pca.explained_variance_ratio_ * 100
    with plt.rc_context({**STYLE, "axes.grid.axis": "both"}):
        fig, ax = plt.subplots(figsize=(7.4, 6.0))
        for cond in conditions:
            g = scores[scores["condition"] == cond]
            ax.scatter(g[a], g[b], s=14, color=palette[cond], alpha=0.5, linewidths=0,
                       label=cond)
        out = scores.loc[outlier_order(flags)]
        ax.scatter(out[a], out[b], s=120, facecolors="none", edgecolors="#d1495b",
                   linewidths=1.8, label=f"outlier (n = {len(out)})")
        for k, (_, r) in enumerate(out.iterrows(), start=1):
            ax.annotate(str(k), (r[a], r[b]), xytext=(7, 5), textcoords="offset points",
                        fontsize=8, fontweight="bold", color="#b23a48")
        ax.set_xlabel(f"{a} ({var[int(a[2:]) - 1]:.1f}%)")
        ax.set_ylabel(f"{b} ({var[int(b[2:]) - 1]:.1f}%)")
        ax.set_title("Multivariate outliers (numbers as in the table)", pad=10)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
        sns.despine(ax=ax)
        fig.tight_layout()
    return fig
