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
    """(condition, baseline) pairs: each vs the reference ("vs_reference"),
    every pair ("all_pairs"), or an explicit list of (condition, baseline)
    tuples, e.g. [("MGS3", "WT"), ("WT, Apt.3", "WT, NC")]."""
    conditions = list(conditions)
    if isinstance(mode, (list, tuple)):
        pairs = [tuple(map(str, p)) for p in mode]
        unknown = sorted({c for p in pairs for c in p} - set(map(str, conditions)))
        if unknown:
            raise ValueError(f"unknown condition(s) in COMPARISONS: {unknown}; "
                             f"conditions are {conditions}")
        if any(len(p) != 2 or p[0] == p[1] for p in pairs):
            raise ValueError("each comparison must be a pair of two different conditions")
        return pairs
    if mode == "vs_reference":
        reference = conditions[0] if reference is None else reference
        if reference not in conditions:
            raise ValueError(f"reference {reference!r} is not one of {conditions}")
        return [(c, reference) for c in conditions if c != reference]
    if mode == "all_pairs":
        return [(b, a) for a, b in combinations(conditions, 2)]
    raise ValueError('mode must be "vs_reference", "all_pairs" or a list of pairs')


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


BRACKET_IN = 0.27      # height of one bracket level, in inches


def _bracket_items(comparisons, conditions, q_col, show_ns):
    """(x positions, stars) of the brackets to draw for one metric."""
    items = []
    for _, r in comparisons.iterrows():
        if r["condition"] not in conditions or r["baseline"] not in conditions:
            continue
        s = stars(r[q_col])
        if not s or (s == "ns" and not show_ns):
            continue
        items.append(((conditions.index(r["baseline"]), conditions.index(r["condition"])), s))
    return items


def _bracket_levels(pairs):
    """Stack comparison brackets: short spans low, no overlap on a level.
    `pairs` = list of (i, j) x positions; returns a level per pair."""
    order = sorted(range(len(pairs)), key=lambda k: (abs(pairs[k][1] - pairs[k][0]), min(pairs[k])))
    taken, levels = {}, [0] * len(pairs)
    for k in order:
        lo, hi = sorted(pairs[k])
        lvl = 0
        while any(lo < b and a < hi for a, b in taken.get(lvl, [])):
            lvl += 1
        taken.setdefault(lvl, []).append((lo, hi))
        levels[k] = lvl
    return levels


def _draw_brackets(ax, items, log):
    """Raise the y axis to make room, then draw the brackets with their stars."""
    from matplotlib import transforms
    fig_h_in = ax.get_position().height * ax.figure.get_figheight()
    step = BRACKET_IN / fig_h_in                      # one level, in axes fraction
    levels = _bracket_levels([p for p, _ in items]) if items else []
    n_lev = (max(levels) + 1) if levels else 0
    data_frac = max(0.35, 1 - 0.04 - step * max(n_lev, 0.6) - 0.02)
    lo, hi = ax.get_ylim()
    if log:
        llo, lhi = np.log10(lo), np.log10(hi)
        ax.set_ylim(lo, 10 ** (llo + (lhi - llo) / data_frac))
    else:
        ax.set_ylim(lo, lo + (hi - lo) / data_frac)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    y0 = data_frac + 0.03
    tick = min(0.022, step * 0.25)
    for ((i, j), s), lvl in zip(items, levels):
        a, b = sorted((i, j))
        y = y0 + lvl * step
        sig = s != "ns"
        colour = "#222222" if sig else "#b3b3b3"
        ax.plot([a + 0.06, a + 0.06, b - 0.06, b - 0.06], [y - tick, y, y, y - tick],
                transform=trans, color=colour, linewidth=1.1 if sig else 0.8,
                clip_on=False, zorder=5, solid_capstyle="butt")
        ax.text((a + b) / 2, y + (0.002 if sig else 0.006), s, transform=trans,
                ha="center", va="center_baseline" if sig else "bottom", clip_on=False, zorder=6,
                **({**STAR_STYLE, "fontsize": 13} if sig else {**NS_STYLE, "fontsize": 8}))


def plot_superplots(df, metrics, labels=None, log_metrics=(), ncols=4,
                    height=3.4, results=None, q_col="q_replicates", title=None,
                    names=None, show_ns="auto"):
    """SuperPlot grid: one panel per metric.

    grey shape : distribution of all cells of the condition (violin)
    small dots : cells, coloured by replicate (by condition if one replicate)
    big dots   : replicate medians (black edge)
    black bar  : mean of the replicate medians; error bar = SD between replicates
    brackets   : one bracket per comparison in `results` (condition vs its
                 baseline), stacked above the data, with the q-value as stars:
                 * < 0.05, ** < 0.01, *** < 0.001; ns in grey (hidden with
                 show_ns=False; "auto" = shown when a panel has at most 3
                 comparisons)
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

    # Brackets per metric; panels grow with the number of bracket levels
    brackets, max_levels = {}, 0
    if results is not None:
        for metric in metrics:
            comp = results[results["metric"] == metric]
            ns_on = (len(comp) <= 3) if show_ns == "auto" else bool(show_ns)
            items = _bracket_items(comp, conditions, q_col, ns_on)
            brackets[metric] = items
            if items:
                max_levels = max(max_levels, max(_bracket_levels([p for p, _ in items])) + 1)
        height = height + BRACKET_IN * max_levels
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
            ax.set_title(pretty(metric, names), pad=10)
            ax.set_ylabel(labels.get(metric, ""))
            sns.despine(ax=ax)

            if results is not None:
                brackets_to_draw = (ax, brackets.get(metric, []), log)
                ax._brackets = brackets_to_draw

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
        for ax in axes.flat:
            if hasattr(ax, "_brackets"):
                _draw_brackets(*ax._brackets)
    return fig


def plot_pairwise_matrices(results, metrics=None, conditions=None, q_col="q_replicates",
                           names=None, ncols=4):
    """One small condition x condition matrix per metric.

    Cell (row A, column B) = Cliff's delta of A vs B (red: A higher than B),
    with stars from the q-value; blank = pair not compared. Shows every
    pairwise comparison at once, also for many conditions.
    """
    res = results
    metrics = [m for m in (metrics or list(dict.fromkeys(res["metric"]))) if m in set(res["metric"])]
    if conditions is None:
        conditions = list(dict.fromkeys(list(res["baseline"].astype(str)) + list(res["condition"].astype(str))))
    conditions = [str(c) for c in conditions]
    n = len(conditions)
    pos = {c: i for i, c in enumerate(conditions)}
    ncols = min(ncols, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    cell = 0.48 if n > 3 else 0.62
    side = cell * n + 1.2
    show_numbers = n <= 6
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(nrows, ncols, figsize=(side * ncols + 1.2, side * nrows + 0.6),
                                 squeeze=False)
        im = None
        for ax, metric in zip(axes.flat, metrics):
            M = np.full((n, n), np.nan)
            Q = np.full((n, n), np.nan)
            for _, r in res[res["metric"] == metric].iterrows():
                a, b = pos.get(str(r["condition"])), pos.get(str(r["baseline"]))
                if a is None or b is None:
                    continue
                d = r["cliffs_delta"]
                # keep the lower triangle: row = later condition
                if a > b:
                    M[a, b], Q[a, b] = d, r[q_col]
                else:
                    M[b, a], Q[b, a] = -d, r[q_col]
            used_rows = [i for i in range(n) if np.isfinite(M[i]).any()]
            used_cols = [j for j in range(n) if np.isfinite(M[:, j]).any()]
            if not used_rows:
                ax.axis("off")
                continue
            sub = M[np.ix_(used_rows, used_cols)]
            subq = Q[np.ix_(used_rows, used_cols)]
            im = ax.imshow(sub, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            for (y, x), d in np.ndenumerate(sub):
                if not np.isfinite(d):
                    continue
                st = stars(subq[y, x])
                ink = "white" if abs(d) > 0.6 else "#111111"
                if st and st != "ns":
                    if show_numbers:
                        ax.text(x, y - 0.17, f"{d:+.2f}", ha="center", va="center", fontsize=7, color=ink)
                    ax.text(x, y + (0.2 if show_numbers else 0.05), st, ha="center", va="center",
                            fontsize=11, fontweight="bold", color=ink)
                elif show_numbers:
                    ax.text(x, y, f"{d:+.2f}", ha="center", va="center", fontsize=7,
                            color=ink, alpha=0.6)
            ax.set_xticks(range(len(used_cols)))
            ax.set_xticklabels([conditions[j] for j in used_cols], rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(len(used_rows)))
            ax.set_yticklabels([conditions[i] for i in used_rows], fontsize=8)
            ax.set_xticks(np.arange(-0.5, len(used_cols)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(used_rows)), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="both", length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(pretty(metric, names), fontsize=10)
        for ax in list(axes.flat)[len(metrics):]:
            ax.axis("off")
        fig.tight_layout(rect=(0, 0, 0.93, 0.96))
        if im is not None:
            cax = fig.add_axes([0.945, 0.25, 0.012, 0.5])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("Cliff's δ (row vs column)", fontsize=8)
            cb.outline.set_visible(False)
        level = "replicate" if q_col == "q_replicates" else "cell"
        fig.text(0.5, 0.995, f"All pairwise comparisons  ·  stars: {level}-level q-value",
                 ha="center", va="top", fontsize=13, fontweight="bold")
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
