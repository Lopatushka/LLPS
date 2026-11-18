"""UMAP map and cell-state clusters of per-cell foci metrics.

Used by UMAP_v2.ipynb. Feature preparation, loading, design checks, names and
plot style are shared with PCA_v2_utils.py / METRICS_utils.py.

Principles:
* UMAP is a map for looking, not for measuring: distances and densities on it
  are distorted, so nothing is tested on UMAP coordinates.
* Clusters are found in the original (z-scored) feature space and only drawn
  on the UMAP, so they don't depend on the UMAP settings.
* Differences between conditions in cluster composition are tested on
  replicate fractions (n = replicates); cell-level tests are descriptive.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score, adjusted_rand_score
from statsmodels.stats.multitest import multipletests

from PCA_v2_utils import (  # noqa: F401  (re-exported for the notebook)
    Path, load_conditions, design_table, check_design, apply_min_foci, missing_table,
    comparison_pairs, prepare_features, plot_feature_distributions, condition_palette,
    stars, pretty, STYLE, STAR_STYLE, NS_STYLE,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import umap


def _quiet_umap(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return umap.UMAP(**kw)


# =====================================================
# 1. UMAP
# =====================================================

def run_umap(Z, n_neighbors=15, min_dist=0.3, seed=42):
    """2D UMAP embedding of the z-scored features (array n x 2)."""
    n_neighbors = int(min(n_neighbors, len(Z) - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _quiet_umap(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed,
                           n_components=2).fit_transform(Z.to_numpy())


def add_embedding(data, emb):
    out = data.copy()
    out["UMAP1"], out["UMAP2"] = emb[:, 0], emb[:, 1]
    return out


def trust(Z, emb, k=10):
    """Trustworthiness (0-1): how well local neighbourhoods are preserved.
    > 0.9 good, 0.8-0.9 acceptable, < 0.8 the map distorts neighbourhoods."""
    return float(trustworthiness(Z.to_numpy(), emb, n_neighbors=int(min(k, len(Z) // 2 - 1))))


def _dot(n, base=1500, lo=5, hi=45):
    """Marker size that suits the number of cells."""
    return float(np.clip(base / max(n, 1), lo, hi))


def _style_map(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    if title:
        ax.set_title(title)


def plot_umap_sweep(Z, conditions, palette, neighbors=(5, 15, 30, 50), min_dists=(0.1, 0.5),
                    seed=42):
    """Grid of UMAPs for several n_neighbors (columns) and min_dist (rows),
    with trustworthiness. Returns (fig, table)."""
    rows = []
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(len(min_dists), len(neighbors),
                                 figsize=(3.3 * len(neighbors), 3.2 * len(min_dists)), squeeze=False)
        cond = np.asarray(conditions)
        for i, md_ in enumerate(min_dists):
            for j, nn in enumerate(neighbors):
                ax = axes[i, j]
                nn_eff = int(min(nn, len(Z) - 1))
                emb = run_umap(Z, nn_eff, md_, seed)
                t = trust(Z, emb)
                rows.append(dict(n_neighbors=nn_eff, min_dist=md_, trustworthiness=round(t, 3)))
                for c in palette:
                    m = cond == c
                    ax.scatter(emb[m, 0], emb[m, 1], s=_dot(len(emb), 800), color=palette[c],
                               alpha=0.7, linewidths=0)
                _style_map(ax, f"n_neighbors = {nn_eff}, min_dist = {md_}")
                ax.title.set_fontsize(9)
                ax.text(0.02, 0.02, f"trust. {t:.2f}", transform=ax.transAxes, fontsize=8,
                        color="#555555")
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=str(k))
                   for k, c in palette.items()]
        fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 8), frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, pd.DataFrame(rows)


def plot_seed_stability(Z, conditions, palette, n_neighbors, min_dist, seeds=(1, 2, 3)):
    """The same UMAP with different random seeds: structures that appear in
    every panel are real, structures in only one are artefacts."""
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(1, len(seeds), figsize=(3.6 * len(seeds), 3.4))
        cond = np.asarray(conditions)
        for ax, s in zip(np.atleast_1d(axes), seeds):
            emb = run_umap(Z, n_neighbors, min_dist, s)
            for c in palette:
                m = cond == c
                ax.scatter(emb[m, 0], emb[m, 1], s=_dot(len(emb), 800), color=palette[c],
                           alpha=0.7, linewidths=0)
            _style_map(ax, f"seed {s}")
        fig.tight_layout()
    return fig


def plot_umap(emb_df, palette, title="UMAP of cells", show_density=True):
    """All cells coloured by condition, with density contours per condition."""
    conditions = list(emb_df["condition"].cat.categories)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(7.4, 6.2))
        for cond in conditions:
            g = emb_df[emb_df["condition"] == cond]
            ax.scatter(g["UMAP1"], g["UMAP2"], s=_dot(len(emb_df), 3000, 8, 60), color=palette[cond],
                       alpha=0.6, linewidths=0, label=f"{cond}  (n = {len(g)})", zorder=2)
            if show_density and len(g) >= 15:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sns.kdeplot(x=g["UMAP1"], y=g["UMAP2"], levels=[0.5], color=palette[cond],
                                linewidths=1.6, ax=ax, zorder=3)
        _style_map(ax, title)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=1.6)
        fig.tight_layout()
    return fig


def plot_umap_facets(emb_df, palette, ncols=3):
    """One panel per condition: its cells in colour, all others in grey."""
    conditions = list(emb_df["condition"].cat.categories)
    ncols = min(ncols, len(conditions))
    nrows = int(np.ceil(len(conditions) / ncols))
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, cond in zip(axes.flat, conditions):
            own = emb_df[emb_df["condition"] == cond]
            other = emb_df[emb_df["condition"] != cond]
            size = _dot(len(emb_df), 1200)
            ax.scatter(other["UMAP1"], other["UMAP2"], s=size * 0.7, color="#dcdcdc", linewidths=0)
            ax.scatter(own["UMAP1"], own["UMAP2"], s=size, color=palette[cond], alpha=0.85, linewidths=0)
            _style_map(ax, f"{cond}  (n = {len(own)})")
        for ax in list(axes.flat)[len(conditions):]:
            ax.axis("off")
        fig.tight_layout()
    return fig


def plot_feature_overlays(emb_df, Z, names=None, ncols=4, extra=None):
    """UMAP coloured by each z-scored feature (clipped at +/-2.5 SD), plus
    optional extra raw columns (e.g. expression, log-coloured)."""
    cols = list(Z.columns) + list(extra or [])
    nrows = int(np.ceil(len(cols) / ncols))
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)
        for ax, col in zip(axes.flat, cols):
            if col in Z.columns:
                v = Z.loc[emb_df.index, col].clip(-2.5, 2.5).to_numpy()
                cmap, vmin, vmax, lab = "RdBu_r", -2.5, 2.5, "z"
            else:
                raw = emb_df[col].astype(float)
                v = np.log10(raw.clip(lower=raw[raw > 0].min())).to_numpy()
                cmap, vmin, vmax, lab = "viridis", np.nanmin(v), np.nanmax(v), "log10"
            order = np.argsort(np.abs(v - (0 if col in Z.columns else np.nanmedian(v))))
            sc = ax.scatter(emb_df["UMAP1"].to_numpy()[order], emb_df["UMAP2"].to_numpy()[order],
                            c=v[order], cmap=cmap, vmin=vmin, vmax=vmax, s=_dot(len(emb_df), 900),
                            edgecolors="#b5b5b5", linewidths=0.3)
            _style_map(ax, pretty(col, names))
            ax.set_xlabel("")
            ax.set_ylabel("")
            cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=7)
            cb.set_label(lab, fontsize=7)
        for ax in list(axes.flat)[len(cols):]:
            ax.axis("off")
        fig.tight_layout()
    return fig


# =====================================================
# 2. CLUSTERS (in feature space)
# =====================================================

def choose_k(Z, k_range=range(2, 9), n_boot=30, seed=0):
    """Quality of KMeans clusterings for each k in the feature space.

    silhouette : how well separated the clusters are (-1..1; > 0.25 some
                 structure, > 0.5 clear clusters).
    stability  : mean adjusted Rand index between the clustering of the full
                 data and of bootstrap resamples (1 = always the same
                 clusters; < 0.6 unstable).
    """
    X = Z.to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for k in k_range:
        if k >= len(X):
            break
        full = KMeans(k, n_init=20, random_state=seed).fit(X)
        sil = silhouette_score(X, full.labels_) if len(set(full.labels_)) > 1 else np.nan
        aris = []
        for _ in range(n_boot):
            idx = rng.integers(len(X), size=len(X))
            km = KMeans(k, n_init=5, random_state=int(rng.integers(1e9))).fit(X[idx])
            aris.append(adjusted_rand_score(full.labels_, km.predict(X)))
        rows.append(dict(k=k, silhouette=sil, stability=float(np.mean(aris)),
                         smallest_cluster_pct=100 * np.bincount(full.labels_).min() / len(X)))
    return pd.DataFrame(rows)


def suggest_k(quality, min_stability=0.75, min_size_pct=5, min_silhouette=0.25):
    """Pick k and say how much to trust it. Returns (k, message).

    Among clusterings without tiny clusters (smallest >= min_size_pct % of
    cells): the highest-silhouette k that is also stable; if none is stable,
    the highest-silhouette k anyway, with a warning that the cells don't form
    distinct clusters.
    """
    sized = quality[quality["smallest_cluster_pct"] >= min_size_pct]
    if sized.empty:
        sized = quality
    stable = sized[sized["stability"] >= min_stability]
    pick = (stable if len(stable) else sized).sort_values("silhouette", ascending=False).iloc[0]
    k = int(pick["k"])
    if len(stable) and pick["silhouette"] >= min_silhouette:
        msg = (f"k = {k}: stable (ARI {pick['stability']:.2f}) and separated "
               f"(silhouette {pick['silhouette']:.2f}). Cell states are supported by the data.")
    elif len(stable):
        msg = (f"k = {k}: stable (ARI {pick['stability']:.2f}) but weakly separated "
               f"(silhouette {pick['silhouette']:.2f}): reproducible regions of a continuum "
               "rather than distinct cell states.")
    else:
        msg = (f"WARNING: no k gives stable clusters (best ARI {sized['stability'].max():.2f} "
               f"< {min_stability}). The cells form a continuum without distinct states; "
               f"k = {k} (best silhouette) is shown for orientation only. Do not interpret the "
               "clusters as cell types.")
    return k, msg


def plot_k_quality(quality, chosen):
    with plt.rc_context(STYLE):
        fig, ax1 = plt.subplots(figsize=(6.4, 3.8))
        ax1.plot(quality["k"], quality["silhouette"], "-o", color="#3b75af", label="silhouette")
        ax1.plot(quality["k"], quality["stability"], "-s", color="#d1495b", label="stability (ARI)")
        ax1.axvline(chosen, color="#999999", linestyle=":", linewidth=1.4)
        ax1.text(chosen, 1.02, f" k = {chosen}", transform=ax1.get_xaxis_transform(), fontsize=9,
                 color="#555555")
        ax1.axhline(0.75, color="#d1495b", linewidth=0.8, linestyle="--", alpha=0.5)
        ax1.set_xlabel("number of clusters k")
        ax1.set_ylabel("score")
        ax1.set_ylim(min(0, quality["silhouette"].min() - 0.05), 1.05)
        ax1.set_xticks(quality["k"])
        ax1.legend(frameon=False, loc="lower left")
        ax1.set_title("How many cell states?", pad=14)
        sns.despine(ax=ax1)
        fig.tight_layout()
    return fig


def fit_clusters(Z, k, seed=0):
    """KMeans in feature space; clusters renumbered 1..k by size (1 = largest)."""
    km = KMeans(k, n_init=50, random_state=seed).fit(Z.to_numpy())
    order = np.argsort(-np.bincount(km.labels_))
    remap = {old: new + 1 for new, old in enumerate(order)}
    return pd.Series([remap[l] for l in km.labels_], index=Z.index, name="cluster")


def cluster_palette(k):
    return dict(zip(range(1, k + 1), sns.color_palette("Set2", k)))


def plot_clusters_on_umap(emb_df, clusters, cpal):
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        for c in sorted(cpal):
            g = emb_df[clusters == c]
            ax.scatter(g["UMAP1"], g["UMAP2"], s=_dot(len(emb_df), 3000, 8, 60), color=cpal[c],
                       alpha=0.8, linewidths=0, label=f"cluster {c}  ({len(g)} cells)")
            ax.text(g["UMAP1"].median(), g["UMAP2"].median(), str(c), fontsize=15, fontweight="bold",
                    ha="center", va="center", color="black",
                    bbox=dict(boxstyle="circle,pad=0.25", facecolor="white", edgecolor=cpal[c],
                              linewidth=2, alpha=0.9))
        _style_map(ax, "Cell states (clusters found in feature space)")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=1.6)
        fig.tight_layout()
    return fig


def cluster_profiles(Z, clusters):
    """Median z-score of each feature per cluster (features x clusters)."""
    return Z.groupby(clusters).median().T


def plot_cluster_profiles(profiles, cpal, names=None):
    """Heatmap: what distinguishes each cluster (red = higher than average)."""
    P = profiles.copy()
    order = P.abs().max(axis=1).sort_values(ascending=False).index
    P = P.loc[order]
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(2.8 + 1.0 * P.shape[1], 0.42 * len(P) + 1.6))
        sns.heatmap(P, cmap="RdBu_r", vmin=-1.5, vmax=1.5, center=0, annot=True, fmt="+.1f",
                    annot_kws={"fontsize": 8}, linewidths=1.5, linecolor="white",
                    cbar_kws={"label": "median z-score in cluster", "shrink": 0.7},
                    yticklabels=[pretty(f, names) for f in P.index],
                    xticklabels=[f"cluster {c}" for c in P.columns], ax=ax)
        for tick, c in zip(ax.get_xticklabels(), P.columns):
            tick.set_color(cpal[c])
            tick.set_fontweight("bold")
        ax.tick_params(length=0)
        ax.set_xlabel("")
        ax.set_title("What characterises each cell state", pad=10)
        fig.tight_layout()
    return fig


def describe_clusters(profiles, names=None, n=3, threshold=0.5):
    """One line per cluster: its most distinctive features, high and low."""
    rows = []
    for c in profiles.columns:
        col = profiles[c]
        hi = col[col >= threshold].sort_values(ascending=False).head(n)
        lo = col[col <= -threshold].sort_values().head(n)
        parts = [f"high {pretty(f, names).lower()} ({v:+.1f})" for f, v in hi.items()]
        parts += [f"low {pretty(f, names).lower()} ({v:+.1f})" for f, v in lo.items()]
        rows.append(dict(cluster=c, description=", ".join(parts) or "close to average on all features"))
    return pd.DataFrame(rows)


def typical_cells(emb_df, Z, clusters, n=5):
    """The n cells closest to each cluster centre (feature space): open their
    images to see what the cluster looks like."""
    rows = []
    for c in sorted(clusters.unique()):
        idx = clusters.index[clusters == c]
        X = Z.loc[idx]
        d = np.linalg.norm(X - X.mean(), axis=1)
        for rank, i in enumerate(idx[np.argsort(d)][:n], start=1):
            r = emb_df.loc[i]
            rows.append(dict(cluster=c, rank=rank, condition=r["condition"], replicate=r["replicate"],
                             image_id=r.get("image_id", i), distance_to_centre=round(float(np.sort(d)[rank - 1]), 2)))
    return pd.DataFrame(rows)


# =====================================================
# 3. CLUSTER COMPOSITION OF CONDITIONS
# =====================================================

def composition(emb_df, clusters):
    """Fraction of each replicate's cells in each cluster (long format)."""
    t = pd.crosstab([emb_df["condition"], emb_df["replicate"]], clusters, normalize="index")
    t = t.stack().rename("fraction").reset_index()
    t.columns = ["condition", "replicate", "cluster", "fraction"]
    return t


def composition_tests(emb_df, clusters, pairs):
    """Per cluster and comparison: is the share of cells in the cluster
    different?

    p_replicates : Welch t-test on replicate fractions (needs >= 2 replicates
                   per condition); the test for conclusions.
    p_cells      : Fisher exact test on cell counts (in cluster vs not);
                   descriptive.
    q_*          : Benjamini-Hochberg over clusters x comparisons.
    Plus the overall chi-square test of conditions x clusters (cells).
    """
    comp = composition(emb_df, clusters)
    rows = []
    for c in sorted(clusters.unique()):
        inc = clusters == c
        for a, b in pairs:
            fa = comp[(comp["condition"] == a) & (comp["cluster"] == c)]["fraction"]
            fb = comp[(comp["condition"] == b) & (comp["cluster"] == c)]["fraction"]
            ca = emb_df["condition"] == a
            cb = emb_df["condition"] == b
            table = [[int((inc & ca).sum()), int((~inc & ca).sum())],
                     [int((inc & cb).sum()), int((~inc & cb).sum())]]
            row = dict(cluster=c, condition=a, baseline=b,
                       pct=100 * table[0][0] / max(sum(table[0]), 1),
                       pct_baseline=100 * table[1][0] / max(sum(table[1]), 1),
                       n_rep=len(fa), n_rep_baseline=len(fb),
                       p_cells=stats.fisher_exact(table).pvalue)
            row["difference_pct"] = row["pct"] - row["pct_baseline"]
            row["p_replicates"] = (stats.ttest_ind(fa, fb, equal_var=False).pvalue
                                   if len(fa) >= 2 and len(fb) >= 2 and (fa.std() + fb.std()) > 0
                                   else np.nan)
            rows.append(row)
    res = pd.DataFrame(rows)
    for p in ("p_replicates", "p_cells"):
        q = p.replace("p_", "q_")
        res[q] = np.nan
        ok = res[p].notna()
        if ok.any():
            res.loc[ok, q] = multipletests(res.loc[ok, p], method="fdr_bh")[1]
    chi = stats.chi2_contingency(pd.crosstab(emb_df["condition"], clusters))
    overall = dict(chi2=chi.statistic, dof=chi.dof, p=chi.pvalue)
    return res, overall


def plot_composition(emb_df, clusters, cpal, palette, results=None, q_col="q_replicates"):
    """Left: stacked bars of cluster shares per condition. Right: one panel per
    cluster with the share per replicate (dots) and its mean (bar), stars vs
    the baseline."""
    comp = composition(emb_df, clusters)
    conditions = list(emb_df["condition"].cat.categories)
    ks = sorted(cpal)
    means = comp.groupby(["condition", "cluster"], observed=True)["fraction"].mean().unstack().reindex(conditions)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(3.4 + 2.6 * len(ks), 4.2))
        gs = fig.add_gridspec(1, len(ks) + 1, width_ratios=[1.4] + [1] * len(ks), wspace=0.45)
        ax = fig.add_subplot(gs[0])
        bottom = np.zeros(len(conditions))
        for c in ks:
            vals = 100 * means[c].fillna(0).to_numpy()
            ax.bar(range(len(conditions)), vals, bottom=bottom, color=cpal[c], width=0.7,
                   edgecolor="white", linewidth=1, label=f"cluster {c}")
            for i, (v, b0) in enumerate(zip(vals, bottom)):
                if v >= 8:
                    ax.text(i, b0 + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8)
            bottom += vals
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=30 if max(map(len, map(str, conditions))) > 6 else 0,
                           ha="right" if max(map(len, map(str, conditions))) > 6 else "center")
        ax.set_ylabel("cells (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Cluster shares")
        ax.tick_params(axis="x", length=0)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2)
        sns.despine(ax=ax)

        rng = np.random.default_rng(0)
        for j, c in enumerate(ks):
            axc = fig.add_subplot(gs[j + 1])
            for i, cond in enumerate(conditions):
                f = 100 * comp[(comp["condition"] == cond) & (comp["cluster"] == c)]["fraction"]
                axc.bar(i, f.mean(), width=0.62, color=palette[cond], alpha=0.35)
                axc.scatter(i + rng.uniform(-0.12, 0.12, len(f)), f, s=36, color=palette[cond],
                            edgecolor="black", linewidths=0.8, zorder=3)
            top = max(5, 100 * comp[comp["cluster"] == c]["fraction"].max())
            axc.set_ylim(0, top * 1.28)
            axc.set_yticks([t for t in axc.get_yticks() if 0 <= t <= min(100, top * 1.05)])
            if results is not None:
                for _, r in results[results["cluster"] == c].iterrows():
                    s = stars(r[q_col])
                    if s:
                        axc.text(conditions.index(r["condition"]), top * 1.14, s, ha="center",
                                 va="center", **(NS_STYLE if s == "ns" else STAR_STYLE))
            axc.set_xticks(range(len(conditions)))
            axc.set_xticklabels(conditions, rotation=30 if max(map(len, map(str, conditions))) > 6 else 0,
                                ha="right" if max(map(len, map(str, conditions))) > 6 else "center")
            axc.tick_params(axis="x", length=0)
            axc.set_title(f"cluster {c}", color=cpal[c], fontweight="bold")
            if j == 0:
                axc.set_ylabel("cells in cluster (%)")
            sns.despine(ax=axc)
        fig.subplots_adjust(top=0.88, bottom=0.28, left=0.06, right=0.98)
    return fig
