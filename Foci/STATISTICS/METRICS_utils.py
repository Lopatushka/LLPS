"""Per-metric comparison of conditions: SuperPlots and replicate-aware tests.

Used by METRICS.ipynb. Input: the aggregated_data.csv of each condition and
replicate (one row per cell, written by agg.py).

Statistics follow the SuperPlot approach (Lord et al., J Cell Biol 2020):
cells from one replicate are not independent, so the test that decides
significance is done on replicate medians (n = number of replicates). The
cell-level test is reported too, but only as a descriptive value.
"""

from itertools import combinations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests


# =====================================================
# 1. LOADING AND DESIGN
# =====================================================

def load_conditions(data):
    """Concatenate the aggregated_data.csv files listed in `data`.

    `data` is a list of dicts with keys "condition", "path" and optionally
    "replicate" (default "1"). Adds the columns `condition` (ordered as in
    `data`) and `replicate` (string) in front of the agg.py columns.
    """
    frames = []
    for item in data:
        df = pd.read_csv(item["path"])
        df.insert(0, "condition", item["condition"])
        df.insert(1, "replicate", str(item.get("replicate", "1")))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    order = list(dict.fromkeys(item["condition"] for item in data))
    df["condition"] = pd.Categorical(df["condition"], categories=order, ordered=True)
    return df


def design_table(df):
    """Cells per condition x replicate, plus totals and cells without foci."""
    counts = df.pivot_table(index="condition", columns="replicate", values="image_id",
                            aggfunc="count", observed=True, fill_value=0)
    counts.columns = [f"rep {c}" for c in counts.columns]
    grouped = df.groupby("condition", observed=True)
    counts["n_replicates"] = grouped["replicate"].nunique()
    counts["n_cells"] = grouped.size()
    counts["cells_without_foci"] = grouped["n_foci"].apply(lambda s: int((s == 0).sum()))
    return counts


def check_design(df, min_replicates=3):
    """Print warnings about the experimental design; return the test level used.

    Returns "replicates" if every condition has >= 2 replicates (replicate-level
    test possible), otherwise "cells".
    """
    n_rep = df.groupby("condition", observed=True)["replicate"].nunique()
    if (n_rep < 2).any():
        print("WARNING: some conditions have only one replicate: "
              f"{', '.join(map(str, n_rep[n_rep < 2].index))}.\n"
              "  Replicate-level tests are impossible. Only cell-level tests are\n"
              "  computed: their p-values treat every cell as independent and\n"
              "  overstate significance. Use them as descriptive only.")
        return "cells"
    if (n_rep < min_replicates).any():
        print(f"WARNING: some conditions have fewer than {min_replicates} replicates: "
              f"{', '.join(map(str, n_rep[n_rep < min_replicates].index))}.\n"
              "  Replicate-level tests have very little power.")
    else:
        print(f"Design OK: every condition has >= {min_replicates} replicates.")
    return "replicates"


def apply_min_foci(df, min_foci):
    """Set a metric to NaN in cells with fewer foci than needed for it.

    `min_foci` maps metric -> minimum n_foci, e.g. {"size_gini": 4}.
    """
    df = df.copy()
    for metric, k in min_foci.items():
        if metric in df.columns:
            df.loc[df["n_foci"] < k, metric] = np.nan
    return df


def missing_table(df, metrics):
    """Fraction of cells with no value (NaN) per metric and condition."""
    present = [m for m in metrics if m in df.columns]
    return (df.groupby("condition", observed=True)[present]
              .agg(lambda s: s.isna().mean())
              .T.round(2))


# =====================================================
# 2. EFFECT SIZES
# =====================================================

def cliffs_delta(x, y):
    """Cliff's delta = P(x > y) - P(x < y), from -1 to 1. 0 = no difference.

    Rank-based: insensitive to outliers and to the scale of the metric.
    Conventional thresholds for |delta|: 0.15 small, 0.33 medium, 0.47 large.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic  # U of x
    return 2 * u / (len(x) * len(y)) - 1


def _replicate_arrays(df, condition, metric):
    sub = df[df["condition"] == condition]
    return [g[metric].dropna().to_numpy(float)
            for _, g in sub.groupby("replicate") if g[metric].notna().any()]


def hierarchical_bootstrap_ratio(arrays_a, arrays_b, n_boot=1000, ci=95, seed=0):
    """CI of median(a) / median(b) by two-level bootstrap.

    Each iteration resamples replicates with replacement, then cells within each
    chosen replicate, so the CI includes replicate-to-replicate variability.
    With one replicate it reduces to an ordinary cell bootstrap.
    """
    rng = np.random.default_rng(seed)

    def resample(arrays):
        picks = rng.integers(len(arrays), size=len(arrays))
        return np.median(np.concatenate(
            [rng.choice(arrays[i], size=len(arrays[i])) for i in picks]))

    ratios = np.empty(n_boot)
    for k in range(n_boot):
        mb = resample(arrays_b)
        ratios[k] = resample(arrays_a) / mb if mb != 0 else np.nan
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) < n_boot / 2:  # reference median often 0: ratio meaningless
        return np.nan, np.nan
    half = (100 - ci) / 2
    return tuple(np.percentile(ratios, [half, 100 - half]))


# =====================================================
# 3. TESTS
# =====================================================

def comparison_pairs(conditions, reference=None, mode="vs_reference"):
    """(condition, baseline) pairs: each vs the reference, or all pairs."""
    conditions = list(conditions)
    if mode == "vs_reference":
        reference = conditions[0] if reference is None else reference
        if reference not in conditions:
            raise ValueError(f"reference {reference!r} is not one of {conditions}")
        return [(c, reference) for c in conditions if c != reference]
    if mode == "all_pairs":
        return [(b, a) for a, b in combinations(conditions, 2)]
    raise ValueError('mode must be "vs_reference" or "all_pairs"')


def compare_metrics(df, metrics, pairs, n_boot=1000, seed=0):
    """One row per metric x comparison with medians, effect sizes and p-values.

    p_replicates : Welch t-test on replicate medians (needs >= 2 replicates per
                   condition). The test to use for conclusions.
    p_cells      : Mann-Whitney U on all cells (descriptive: cells of one
                   replicate are not independent).
    q_*          : Benjamini-Hochberg FDR over all metrics x comparisons.
    """
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            warnings.warn(f"{metric} not in the data; skipped")
            continue
        for a, b in pairs:
            xa = df.loc[df["condition"] == a, metric].dropna()
            xb = df.loc[df["condition"] == b, metric].dropna()
            reps_a = _replicate_arrays(df, a, metric)
            reps_b = _replicate_arrays(df, b, metric)
            ra = np.array([np.median(r) for r in reps_a])
            rb = np.array([np.median(r) for r in reps_b])

            row = dict(metric=metric, condition=a, baseline=b,
                       n_rep=len(ra), n_rep_baseline=len(rb),
                       n_cells=len(xa), n_cells_baseline=len(xb),
                       median=xa.median(), median_baseline=xb.median())
            row["ratio"] = (row["median"] / row["median_baseline"]
                            if row["median_baseline"] else np.nan)
            if len(xa) and len(xb):
                row["ratio_ci_low"], row["ratio_ci_high"] = hierarchical_bootstrap_ratio(
                    reps_a, reps_b, n_boot=n_boot, seed=seed)
                row["cliffs_delta"] = cliffs_delta(xa, xb)
                row["p_cells"] = stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue
            else:
                row.update(ratio_ci_low=np.nan, ratio_ci_high=np.nan,
                           cliffs_delta=np.nan, p_cells=np.nan)
            if len(ra) >= 2 and len(rb) >= 2:
                row["p_replicates"] = stats.ttest_ind(ra, rb, equal_var=False).pvalue
            else:
                row["p_replicates"] = np.nan
            rows.append(row)

    res = pd.DataFrame(rows)
    for p in ("p_replicates", "p_cells"):
        q = p.replace("p_", "q_")
        res[q] = np.nan
        ok = res[p].notna()
        if ok.any():
            res.loc[ok, q] = multipletests(res.loc[ok, p], method="fdr_bh")[1]
    return res


def omnibus_tests(df, metrics):
    """Is there any difference between conditions? One row per metric.

    p_anova_replicates : one-way ANOVA on replicate medians.
    p_kruskal_cells    : Kruskal-Wallis on cells (descriptive).
    """
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        groups = [g[metric].dropna() for _, g in df.groupby("condition", observed=True)]
        rep_groups = [g.groupby("replicate")[metric].median().dropna()
                      for _, g in df.groupby("condition", observed=True)]
        row = dict(metric=metric)
        row["p_kruskal_cells"] = (stats.kruskal(*groups).pvalue
                                  if sum(len(g) > 0 for g in groups) >= 2 else np.nan)
        row["p_anova_replicates"] = (stats.f_oneway(*rep_groups).pvalue
                                     if all(len(g) >= 2 for g in rep_groups) else np.nan)
        rows.append(row)
    res = pd.DataFrame(rows)
    for p in ("p_anova_replicates", "p_kruskal_cells"):
        ok = res[p].notna()
        res[p.replace("p_", "q_")] = np.nan
        if ok.any():
            res.loc[ok, p.replace("p_", "q_")] = multipletests(res.loc[ok, p], method="fdr_bh")[1]
    return res


def stars(q):
    """Significance stars for a q-value."""
    if pd.isna(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"


# =====================================================
# 4. PLOTS
# =====================================================

# Names shown on the plots instead of the column names
DISPLAY_NAMES = {
    "n_foci": "Foci per cell",
    "foci_density": "Foci density",
    "area_fraction": "Area fraction",
    "fraction_signal_in_foci": "Signal in foci",
    "partition_coefficient": "Partition coefficient",
    "area": "Focus area",
    "equivalent_diameter_area": "Focus diameter",
    "eccentricity": "Eccentricity",
    "solidity": "Solidity",
    "perimeter": "Focus perimeter",
    "size_gini": "Size inequality (Gini)",
    "clark_evans_ratio": "Clark–Evans ratio",
    "nn_distance": "Nearest-focus distance",
    "nn_gap": "Gap to nearest focus",
    "fraction_touching": "Touching foci",
    "mean_intensity": "Focus mean intensity",
    "max_intensity": "Focus max intensity",
    "min_intensity": "Focus min intensity",
    "nucleus_mean_intensity": "Nuclear intensity",
    "nucleoplasm_mean_intensity": "Nucleoplasm intensity",
}


def pretty(metric, names=None):
    """Display name of a metric: `names`, then DISPLAY_NAMES, then the column
    name with underscores replaced by spaces."""
    names = names or {}
    return names.get(metric) or DISPLAY_NAMES.get(metric) or metric.replace("_", " ").capitalize()


STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.labelcolor": "#333333",
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "xtick.color": "#333333",
    "ytick.color": "#555555",
    "legend.fontsize": 9,
    "figure.facecolor": "white",
}

STAR_STYLE = dict(fontsize=15, fontweight="bold", color="#111111")
NS_STYLE = dict(fontsize=9, style="italic", color="#888888")


def _headroom(ax, log, fraction=0.16):
    """Extend the top of the y axis by `fraction` of its span; return the y
    position for the stars (middle of the added band)."""
    lo, hi = ax.get_ylim()
    if log:
        llo, lhi = np.log10(lo), np.log10(hi)
        new = lhi + (lhi - llo) * fraction
        ax.set_ylim(lo, 10 ** new)
        return 10 ** (lhi + (new - lhi) * 0.45)
    new = hi + (hi - lo) * fraction
    ax.set_ylim(lo, new)
    return hi + (new - hi) * 0.45


def plot_superplots(df, metrics, labels=None, log_metrics=(), ncols=4,
                    height=3.4, results=None, q_col="q_replicates", title=None,
                    names=None):
    """SuperPlot grid: one panel per metric.

    grey shape : distribution of all cells of the condition (violin)
    small dots : cells, coloured by replicate (by condition if one replicate)
    big dots   : replicate medians (black edge)
    black bar  : mean of the replicate medians; error bar = SD between replicates
    stars      : q-value of each comparison vs its baseline, if `results` given,
                 in a band above the data: * < 0.05, ** < 0.01, *** < 0.001
    title      : figure title, shown above the legend
    names      : {metric: panel title}, overrides DISPLAY_NAMES
    labels     : {metric: y-axis label (unit)}
    """
    labels = labels or {}
    metrics = [m for m in metrics if m in df.columns]
    conditions = list(df["condition"].cat.categories)
    replicates = sorted(df["replicate"].unique(), key=str)
    one_rep = len(replicates) == 1
    if one_rep:  # colour by condition: a single replicate colour tells nothing
        cond_palette = dict(zip(conditions, sns.color_palette("colorblind", len(conditions))))
    else:
        rep_palette = dict(zip(replicates, sns.color_palette("colorblind", len(replicates))))
    colour = (lambda cond, rep: cond_palette[cond]) if one_rep else (lambda cond, rep: rep_palette[rep])

    rotate = max(len(str(c)) for c in conditions) > 6 or len(conditions) > 4
    ncols = min(ncols, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    rng = np.random.default_rng(0)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=(ncols * (1.0 + 0.75 * len(conditions)), nrows * height))

        for ax, metric in zip(axes.flat, metrics):
            log = metric in log_metrics and (df[metric].dropna() > 0).all()
            for i, cond in enumerate(conditions):
                sub = df[(df["condition"] == cond) & df[metric].notna()]
                if sub.empty:
                    continue
                all_vals = sub[metric].to_numpy(float)
                if len(all_vals) >= 3 and np.ptp(all_vals) > 0:
                    vdata = np.log10(all_vals) if log else all_vals
                    parts = ax.violinplot(vdata, positions=[i], widths=0.8, showextrema=False)
                    for body in parts["bodies"]:
                        if log:  # violin computed in log space, drawn on the log axis
                            path = body.get_paths()[0]
                            path.vertices[:, 1] = 10 ** path.vertices[:, 1]
                        body.set_facecolor("#ececec")
                        body.set_edgecolor("none")
                        body.set_alpha(1)
                        body.set_zorder(0)

                reps = sorted(sub["replicate"].unique(), key=str)
                offsets = np.linspace(-0.18, 0.18, len(reps)) if len(reps) > 1 else [0.0]
                rep_medians = []
                for off, rep in zip(offsets, reps):
                    vals = sub.loc[sub["replicate"] == rep, metric].to_numpy(float)
                    c = colour(cond, rep)
                    ax.scatter(i + off + rng.uniform(-0.09, 0.09, len(vals)), vals, s=9,
                               color=c, alpha=0.45, linewidths=0, zorder=1)
                    med = np.median(vals)
                    rep_medians.append(med)
                    ax.scatter(i + off, med, s=70, color=c, edgecolor="black",
                               linewidths=1, zorder=4)
                m = np.mean(rep_medians)
                ax.hlines(m, i - 0.3, i + 0.3, color="black", linewidth=2.2, zorder=3)
                if len(rep_medians) > 1:
                    ax.errorbar(i, m, yerr=np.std(rep_medians, ddof=1), color="black",
                                capsize=4, capthick=1.2, linewidth=1.2, zorder=3)

            # y range from the data (the log-space violins would distort autoscaling)
            finite = df[metric].dropna().to_numpy(float)
            if len(finite):
                lo, hi = finite.min(), finite.max()
                if log:
                    llo, lhi = np.log10(lo), np.log10(hi)
                    pad = 0.06 * (lhi - llo or 1)
                    ax.set_ylim(10 ** (llo - pad), 10 ** (lhi + pad))
                else:
                    pad = 0.06 * (hi - lo or abs(hi) or 1)
                    ax.set_ylim(lo - pad, hi + pad)
            if log:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(plt.NullFormatter())
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=30 if rotate else 0,
                               ha="right" if rotate else "center")
            ax.set_xlim(-0.6, len(conditions) - 0.4)
            ax.tick_params(axis="x", length=0)
            ax.set_title(pretty(metric, names), pad=8)
            ax.set_ylabel(labels.get(metric, ""))
            sns.despine(ax=ax)

            if results is not None:
                y_star = _headroom(ax, log)
                sub = results[results["metric"] == metric]
                for _, r in sub.iterrows():
                    s = stars(r[q_col])
                    if s and r["condition"] in conditions:
                        ax.text(conditions.index(r["condition"]), y_star, s,
                                ha="center", va="center",
                                **(NS_STYLE if s == "ns" else STAR_STYLE))

        for ax in list(axes.flat)[len(metrics):]:
            ax.axis("off")

        if one_rep:
            handles = []
        else:
            handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=c,
                                  markeredgecolor="black", label=f"Replicate {r}")
                       for r, c in rep_palette.items()]
        # Fixed band (inches) at the top for the title and the legend
        fig_h = fig.get_figheight()
        title_h = 0.3 if title else 0.0
        band = title_h + (0.22 if handles else 0.0)
        if title:
            # fig.text, not suptitle: tight_layout would reserve space for a suptitle twice
            fig.text(0.5, 1 - 0.06 / fig_h, title, ha="center", va="top",
                     fontsize=14, fontweight="bold")
        if handles:
            fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 8),
                       bbox_to_anchor=(0.5, 1 - (title_h - 0.02) / fig_h), frameon=False,
                       handletextpad=0.3, columnspacing=1.2, borderaxespad=0)
        fig.tight_layout(rect=(0, 0, 1, 1 - band / fig_h), w_pad=1.5, h_pad=1.5)
    return fig


def plot_effect_heatmap(results, q_col="q_replicates", metrics=None, names=None):
    """Cliff's delta per metric x comparison, with significance stars."""
    res = results.copy()
    res["comparison"] = res["condition"].astype(str) + " vs " + res["baseline"].astype(str)
    order = metrics or list(dict.fromkeys(res["metric"]))
    order = [m for m in order if m in set(res["metric"])]
    cols = list(dict.fromkeys(res["comparison"]))
    delta = res.pivot(index="metric", columns="comparison", values="cliffs_delta").reindex(index=order, columns=cols)
    q = res.pivot(index="metric", columns="comparison", values=q_col).reindex(index=order, columns=cols)

    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(2.6 + 1.25 * delta.shape[1], 0.5 * len(order) + 1.6))
        sns.heatmap(delta, cmap="RdBu_r", vmin=-1, vmax=1, center=0, linewidths=1.5,
                    linecolor="white", cbar_kws={"label": "Cliff's δ (condition vs baseline)",
                                                 "shrink": 0.8}, ax=ax)
        for r, metric in enumerate(order):
            for c, comp in enumerate(cols):
                d = delta.loc[metric, comp]
                if pd.isna(d):
                    continue
                ink = "white" if abs(d) > 0.6 else "#111111"
                s = stars(q.loc[metric, comp])
                if s and s != "ns":
                    ax.text(c + 0.5, r + 0.36, f"{d:+.2f}", ha="center", va="center",
                            fontsize=8, color=ink)
                    ax.text(c + 0.5, r + 0.72, s, ha="center", va="center",
                            fontsize=13, fontweight="bold", color=ink)
                else:
                    ax.text(c + 0.5, r + 0.5, f"{d:+.2f}", ha="center", va="center",
                            fontsize=8, color=ink, alpha=0.75)
        ax.set_yticklabels([pretty(m, names) for m in order], rotation=0)
        ax.set_xticklabels(cols, rotation=0)
        ax.tick_params(length=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        level = "replicate" if q_col == "q_replicates" else "cell"
        ax.set_title(f"Effect size  ·  stars: {level}-level q-value", pad=10)
        fig.tight_layout()
    return fig


# =====================================================
# 5. EXPRESSION LEVEL
# =====================================================

def expression_correlation(df, metrics, expr_col="nucleus_mean_intensity"):
    """Spearman rho of each metric with expression level, per condition."""
    rows = []
    for metric in metrics:
        if metric not in df.columns or metric == expr_col:
            continue
        row = dict(metric=metric)
        for cond, g in df.groupby("condition", observed=True):
            g = g[[metric, expr_col]].dropna()
            row[str(cond)] = stats.spearmanr(g[metric], g[expr_col]).statistic if len(g) >= 5 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric").round(2)


def expression_matched(df, expr_col="nucleus_mean_intensity", quantiles=(0.05, 0.95)):
    """Keep only cells whose expression lies in the range shared by all conditions.

    The shared range is from the highest lower quantile to the lowest upper
    quantile across conditions. Returns the filtered table and the range.
    """
    q = df.groupby("condition", observed=True)[expr_col].quantile(list(quantiles)).unstack()
    lo, hi = q.iloc[:, 0].max(), q.iloc[:, 1].min()
    if lo >= hi:
        raise ValueError("The conditions' expression ranges do not overlap: "
                         "an expression-matched comparison is impossible.")
    return df[df[expr_col].between(lo, hi)].copy(), (lo, hi)
