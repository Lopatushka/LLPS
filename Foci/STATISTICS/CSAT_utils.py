"""Saturation concentration (c_sat) from single-cell expression levels.

Used by CSAT.ipynb. Each cell is one point on the phase diagram: its total
nuclear concentration (nucleus_mean_intensity) against measures of the dilute
and dense phase. Three independent estimates:

1. onset      - expression at which half the cells have foci (logistic fit);
                needs cells with and without foci.
2. buffering  - the dilute phase (nucleoplasm intensity) follows the total
                below c_sat and levels off above it; segmented fit in log-log
                space, compared with a straight line (no saturation) by BIC.
3. lever rule - the fraction of protein in foci rises as 1 - c_sat / c_total;
                fit of that curve.

Uncertainty: hierarchical bootstrap (replicates, then cells). Loading, design
checks, names and plot style are shared with METRICS_utils.py.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.stats.multitest import multipletests

from METRICS_utils import (  # noqa: F401  (re-exported for the notebook)
    Path, load_conditions as _load_conditions, design_table, check_design,
    comparison_pairs, stars, pretty, STYLE, STAR_STYLE, NS_STYLE,
)

TOTAL = "nucleus_mean_intensity"
DILUTE = "nucleoplasm_mean_intensity"


def condition_palette(conditions):
    return dict(zip(conditions, sns.color_palette("colorblind", len(conditions))))


# =====================================================
# 1. LOADING
# =====================================================

def load_conditions(data):
    """As METRICS_utils.load_conditions, plus nucleoplasm_mean_intensity.

    Older aggregated_data.csv files lack nucleoplasm_mean_intensity: it is
    then read from summary/summary.csv next to the file (written by
    outsu.py / gmm.py in the same output folder).
    """
    df = _load_conditions(data)
    if DILUTE in df.columns and df[DILUTE].notna().all():
        return df
    parts = []
    for item in data:
        summary = Path(item["path"]).parent / "summary" / "summary.csv"
        if not summary.is_file():
            raise FileNotFoundError(
                f"{DILUTE} is not in {item['path']} and {summary} does not exist. "
                "Rerun agg.py with the current version.")
        s = pd.read_csv(summary, usecols=["image_id", DILUTE])
        s["image_id"] = s["image_id"].astype(str)
        s["condition"] = item["condition"]
        s["replicate"] = str(item.get("replicate", "1"))
        parts.append(s)
    extra = pd.concat(parts, ignore_index=True)
    df = df.drop(columns=[DILUTE], errors="ignore")
    df["image_id"] = df["image_id"].astype(str)
    cats = df["condition"].cat.categories
    df["condition"] = df["condition"].astype(str)
    df = df.merge(extra, on=["condition", "replicate", "image_id"], how="left")
    df["condition"] = pd.Categorical(df["condition"], categories=cats, ordered=True)
    print(f"{DILUTE} read from summary/summary.csv next to each aggregated_data.csv")
    return df


def subtract_offset(df, offset):
    """Subtract a constant intensity offset (camera offset / background) from
    the total and dilute intensities; cells that become <= 0 are dropped."""
    df = df.copy()
    for col in (TOTAL, DILUTE):
        df[col] = df[col] - offset
    bad = (df[TOTAL] <= 0) | (df[DILUTE] <= 0)
    if bad.any():
        print(f"{bad.sum()} cells with intensity <= offset removed")
    return df[~bad]


# =====================================================
# 2. MODELS
# =====================================================

def _segmented_fixed_k(x, y, k):
    """y = a + x (x < k), y = a + k + beta (x - k) (x >= k); OLS for a, beta."""
    above = x >= k
    target = y - np.where(above, k, x)
    design = np.column_stack([np.ones_like(x), np.where(above, x - k, 0.0)])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    sse = float(((target - design @ coef) ** 2).sum())
    return coef[0], coef[1], sse


def fit_buffering(total, dilute, n_grid=60):
    """Segmented fit of log10(dilute) vs log10(total).

    Model: slope 1 below the break k (no foci: dilute = total, up to a
    constant factor a for blur / ring exclusion), slope beta above it.
    beta = 0 is ideal buffering (dilute fixed at c_sat), beta = 1 means no
    buffering. Compared by BIC with a straight line log10(dilute) =
    a + beta log10(total) (no break). Returns a dict with c_sat = 10**k
    (total concentration at the break), beta, delta_bic (> 0 favours the
    break) and in_range (break inside the observed expression range).
    """
    x, y = np.log10(np.asarray(total, float)), np.log10(np.asarray(dilute, float))
    n = len(x)
    if n < 8:
        return dict(c_sat=np.nan, beta=np.nan, beta_line=np.nan, delta_bic=np.nan, in_range=False)
    # straight line
    b_line, a_line = np.polyfit(x, y, 1)
    sse_line = float(((y - (a_line + b_line * x)) ** 2).sum())
    # segmented: break at data quantiles, at least 4 cells on each side
    xs = np.sort(x)
    grid = np.unique(np.quantile(xs[3:-3], np.linspace(0, 1, n_grid))) if n > 10 else xs[3:-3]
    best = min((_segmented_fixed_k(x, y, k) + (k,) for k in grid), key=lambda t: t[2])
    a, beta, sse_seg, k = best
    bic = lambda sse, p: n * np.log(max(sse, 1e-300) / n) + p * np.log(n)
    delta = bic(sse_line, 2) - bic(sse_seg, 3)
    return dict(c_sat=10 ** k, beta=beta, beta_line=b_line, a=a, delta_bic=delta,
                in_range=True, a_line=a_line)


def _lever(x, amplitude, c_sat):
    return np.where(x > c_sat, amplitude * (1 - c_sat / x), 0.0)


def fit_lever(total, fraction):
    """Fit fraction_signal_in_foci = A (1 - c_sat / total) for total > c_sat.

    Lever rule for a fixed dilute concentration c_sat. Returns dict(c_sat, A,
    rho): rho = Spearman correlation of fraction with total; the lever rule
    predicts a clearly positive rho. at_bound = c_sat stuck at the lower
    bound (no threshold found in the data).
    """
    x, f = np.asarray(total, float), np.asarray(fraction, float)
    ok = np.isfinite(x) & np.isfinite(f)
    x, f = x[ok], f[ok]
    rho = stats.spearmanr(x, f).statistic if len(x) >= 5 else np.nan
    if len(x) < 8:
        return dict(c_sat=np.nan, A=np.nan, rho=rho, at_bound=True)
    lo = x.min() * 1e-3
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (A, c), _ = curve_fit(_lever, x, f, p0=[max(f.max(), 1e-3), np.quantile(x, 0.1)],
                                  bounds=([0, lo], [1.5, x.max()]), maxfev=5000)
    except RuntimeError:
        return dict(c_sat=np.nan, A=np.nan, rho=rho, at_bound=True)
    return dict(c_sat=c, A=A, rho=rho, at_bound=c < x.min() * 0.05)


def fit_onset(total, has_foci, min_each=5):
    """Logistic fit of P(foci) vs log10(total); c50 = total at P = 0.5.

    Needs at least `min_each` cells with and without foci, otherwise NaN.
    """
    x = np.log10(np.asarray(total, float))
    y = np.asarray(has_foci, bool)
    if y.sum() < min_each or (~y).sum() < min_each:
        return dict(c50=np.nan, slope=np.nan, n_with=int(y.sum()), n_without=int((~y).sum()))
    import statsmodels.api as sm
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.Logit(y.astype(float), sm.add_constant(x)).fit(disp=0)
        b0, b1 = m.params
        c50 = 10 ** (-b0 / b1) if b1 > 0 else np.nan
    except Exception:
        b1, c50 = np.nan, np.nan
    return dict(c50=c50, slope=b1, n_with=int(y.sum()), n_without=int((~y).sum()))


# =====================================================
# 3. ESTIMATES WITH BOOTSTRAP
# =====================================================

def _boot_frames(g, n_boot, rng):
    groups = [r for _, r in g.groupby("replicate")]
    for _ in range(n_boot):
        picks = rng.integers(len(groups), size=len(groups))
        yield pd.concat([groups[i].iloc[rng.integers(len(groups[i]), size=len(groups[i]))]
                         for i in picks])


def estimate_all(df, presence, n_boot=500, confidence=0.95, seed=0):
    """All three estimates per condition, with hierarchical-bootstrap CIs.

    presence : function(df) -> boolean Series, "this cell has phase separated".
    Returns (table, boots): table with one row per condition x method;
    boots = {(condition, method): array of bootstrap estimates}.
    """
    rng = np.random.default_rng(seed)
    half = (1 - confidence) / 2
    rows, boots = [], {}
    for cond, g in df.groupby("condition", observed=True):
        g = g.dropna(subset=[TOTAL])
        point = dict(
            onset=fit_onset(g[TOTAL], presence(g))["c50"],
            buffering=fit_buffering(g[TOTAL], g[DILUTE])["c_sat"],
            lever=fit_lever(g[TOTAL], g["fraction_signal_in_foci"])["c_sat"],
        )
        bfit = fit_buffering(g[TOTAL], g[DILUTE])
        lfit = fit_lever(g[TOTAL], g["fraction_signal_in_foci"])
        ofit = fit_onset(g[TOTAL], presence(g))
        b = {m: [] for m in point}
        betas = []
        for s in _boot_frames(g, n_boot, rng):
            if not np.isnan(point["onset"]):
                b["onset"].append(fit_onset(s[TOTAL], presence(s))["c50"])
            fb = fit_buffering(s[TOTAL], s[DILUTE])
            b["buffering"].append(fb["c_sat"])
            betas.append(fb["beta"])
            b["lever"].append(fit_lever(s[TOTAL], s["fraction_signal_in_foci"])["c_sat"])
        lo_x, hi_x = g[TOTAL].min(), g[TOTAL].max()
        betas = np.array(betas, float)
        for m, est in point.items():
            arr = np.array(b[m], float)
            arr = arr[np.isfinite(arr)]
            boots[(cond, m)] = arr
            row = dict(condition=cond, method=m, n_cells=len(g), estimate=est,
                       ci_low=np.quantile(arr, half) if len(arr) > 10 else np.nan,
                       ci_high=np.quantile(arr, 1 - half) if len(arr) > 10 else np.nan)
            if m == "buffering":
                row.update(beta=bfit["beta"],
                           beta_ci_low=np.nanquantile(betas, half),
                           beta_ci_high=np.nanquantile(betas, 1 - half),
                           delta_bic=bfit["delta_bic"])
            if m == "lever":
                row.update(rho=lfit["rho"])
            if m == "onset":
                row.update(n_with=ofit["n_with"], n_without=ofit["n_without"])
            row["verdict"] = _verdict(m, row, lo_x, hi_x)
            rows.append(row)
    return pd.DataFrame(rows), boots


def _verdict(method, row, lo_x, hi_x):
    """Plain-language reliability of one estimate."""
    est = row["estimate"]
    if method == "onset":
        if np.isnan(est):
            return (f"not estimable: {row['n_without']} cells without foci "
                    f"(needs >= 5 with and without)")
        return "ok" if lo_x <= est <= hi_x else "outside the expression range: extrapolated"
    if method == "buffering":
        if np.isnan(est):
            return "too few cells"
        if row["beta_ci_low"] > 0.8:
            return (f"no buffering (beta = {row['beta']:.2f}): dilute phase rises with total, "
                    "c_sat not identifiable")
        if row["delta_bic"] < 2:
            return "no clear break (BIC prefers a straight line): unreliable"
        return "ok"
    if method == "lever":
        if np.isnan(est):
            return "fit failed"
        if not (row["rho"] > 0.2):
            return (f"lever rule not followed (signal in foci does not rise with total, "
                    f"rho = {row['rho']:.2f}): c_sat not identifiable")
        return "ok" if est >= lo_x else "below the expression range: extrapolated"
    return ""


def compare_to_reference(table, boots, reference, confidence=0.95):
    """Ratio of each condition's c_sat to the reference, per method, with a
    bootstrap CI (independent bootstraps of the two conditions). Only for
    estimates whose verdict is ok or extrapolated."""
    half = (1 - confidence) / 2
    rows = []
    usable = table["verdict"].str.startswith(("ok", "outside", "below"))
    t = table[usable]
    for method in t["method"].unique():
        ref = t[(t["condition"] == reference) & (t["method"] == method)]
        if ref.empty:
            continue
        rb = boots[(reference, method)]
        for _, r in t[(t["method"] == method) & (t["condition"] != reference)].iterrows():
            cb = boots[(r["condition"], method)]
            n = min(len(cb), len(rb))
            if n < 20:
                continue
            ratio = cb[:n] / rb[:n]
            p = 2 * min((ratio <= 1).mean(), (ratio >= 1).mean())
            rows.append(dict(method=method, condition=r["condition"], baseline=reference,
                             ratio=r["estimate"] / ref["estimate"].iloc[0],
                             ci_low=np.quantile(ratio, half), ci_high=np.quantile(ratio, 1 - half),
                             p_bootstrap=max(p, 1 / n)))
    res = pd.DataFrame(rows, columns=["method", "condition", "baseline", "ratio", "ci_low",
                                      "ci_high", "p_bootstrap"])
    if len(res):
        res["q"] = multipletests(res["p_bootstrap"], method="fdr_bh")[1]
    return res


def replicate_estimates(df, presence, min_cells=20):
    """c_sat per replicate (buffering and lever methods), for a replicate-level
    test; replicates with fewer than `min_cells` cells are skipped."""
    rows = []
    for (cond, rep), g in df.groupby(["condition", "replicate"], observed=True):
        if len(g) < min_cells:
            continue
        rows.append(dict(condition=cond, replicate=rep, n_cells=len(g),
                         buffering=fit_buffering(g[TOTAL], g[DILUTE])["c_sat"],
                         beta=fit_buffering(g[TOTAL], g[DILUTE])["beta"],
                         lever=fit_lever(g[TOTAL], g["fraction_signal_in_foci"])["c_sat"],
                         onset=fit_onset(g[TOTAL], presence(g))["c50"]))
    return pd.DataFrame(rows)


def replicate_tests(rep, pairs):
    """Welch t-test on log10 c_sat of replicates, each condition vs baseline."""
    rows = []
    for method in ("buffering", "lever", "onset", "beta"):
        if method not in rep:
            continue
        for a, b in pairs:
            xa = rep.loc[rep["condition"] == a, method].dropna()
            xb = rep.loc[rep["condition"] == b, method].dropna()
            if len(xa) < 2 or len(xb) < 2:
                continue
            ta, tb = (xa, xb) if method == "beta" else (np.log10(xa), np.log10(xb))
            rows.append(dict(method=method, condition=a, baseline=b, n_rep=len(xa),
                             n_rep_baseline=len(xb),
                             p_replicates=stats.ttest_ind(ta, tb, equal_var=False).pvalue))
    res = pd.DataFrame(rows)
    if len(res):
        res["q_replicates"] = multipletests(res["p_replicates"], method="fdr_bh")[1]
    return res


# =====================================================
# 4. PLOTS
# =====================================================

def _fmt_log(axis):
    """Plain-number labels at 1, 2, 5 x 10^k on a log axis, no minor labels."""
    from matplotlib.ticker import LogLocator, NullFormatter
    axis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _log_axis(ax):
    ax.set_xscale("log")
    _fmt_log(ax.xaxis)


def plot_phase_overview(df, palette, presence):
    """Four views of each cell on the phase diagram (x = total, log)."""
    panels = [
        (DILUTE, "Dilute phase", "nucleoplasm intensity", True),
        ("fraction_signal_in_foci", "Protein in foci", "fraction of nuclear signal", False),
        ("area_fraction", "Area in foci", "fraction of nucleus", False),
        ("partition_coefficient", "Partition coefficient", "focus / nucleoplasm", False),
    ]
    with plt.rc_context({**STYLE, "axes.grid.axis": "both"}):
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
        for ax, (col, title, ylabel, loglog) in zip(axes, panels):
            for cond, g in df.groupby("condition", observed=True):
                has = presence(g)
                ax.scatter(g.loc[has, TOTAL], g.loc[has, col], s=20, color=palette[cond],
                           alpha=0.7, linewidths=0, label=str(cond))
                if (~has).any():
                    ax.scatter(g.loc[~has, TOTAL], g.loc[~has, col], s=22, facecolors="none",
                               edgecolors=palette[cond], linewidths=1)
            _log_axis(ax)
            if loglog:
                ax.set_yscale("log")
                _fmt_log(ax.yaxis)
                lims = [df[[TOTAL, DILUTE]].min().min() * 0.8, df[[TOTAL, DILUTE]].max().max() * 1.2]
                ax.plot(lims, lims, "--", color="#888888", linewidth=1, label="dilute = total")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("total (nuclear intensity)")
            sns.despine(ax=ax)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


def plot_onset(df, palette, presence, n_bins=8):
    """Fraction of cells with foci in expression bins, with the logistic fit."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        edges = np.quantile(np.log10(df[TOTAL]), np.linspace(0, 1, n_bins + 1))
        for cond, g in df.groupby("condition", observed=True):
            lx = np.log10(g[TOTAL])
            has = presence(g)
            idx = np.clip(np.digitize(lx, edges[1:-1]), 0, n_bins - 1)
            frac = pd.Series(has.to_numpy()).groupby(idx).mean()
            mid = pd.Series(lx.to_numpy()).groupby(idx).median()
            ax.plot(10 ** mid, frac, "o", color=palette[cond], markersize=7, label=str(cond))
            fit = fit_onset(g[TOTAL], has)
            if np.isfinite(fit["c50"]):
                xx = np.linspace(edges[0], edges[-1], 200)
                ax.plot(10 ** xx, 1 / (1 + np.exp(-fit["slope"] * (xx - np.log10(fit["c50"])))),
                        color=palette[cond], linewidth=2)
                ax.axvline(fit["c50"], color=palette[cond], linestyle=":", linewidth=1.3)
        _log_axis(ax)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("total (nuclear intensity)")
        ax.set_ylabel("fraction of cells with foci")
        ax.set_title("Onset of phase separation", pad=10)
        ax.legend(frameon=False)
        sns.despine(ax=ax)
        fig.tight_layout()
    return fig


def plot_buffering(df, palette, table):
    """Dilute vs total (log-log) per condition with the segmented fit."""
    conds = list(df["condition"].cat.categories)
    with plt.rc_context({**STYLE, "axes.grid.axis": "both"}):
        fig, axes = plt.subplots(1, len(conds), figsize=(4.6 * len(conds), 4.4),
                                 sharex=True, sharey=True, squeeze=False)
        lims = [df[[TOTAL, DILUTE]].min().min() * 0.8, df[[TOTAL, DILUTE]].max().max() * 1.25]
        for ax, cond in zip(axes.flat, conds):
            g = df[df["condition"] == cond]
            c = palette[cond]
            ax.plot(lims, lims, "--", color="#999999", linewidth=1, zorder=1)
            ax.text(lims[1] / 1.3, lims[1] / 1.3, "dilute = total ", ha="right", va="bottom",
                    fontsize=7.5, color="#888888", rotation=38)
            ax.scatter(g[TOTAL], g[DILUTE], s=20, color=c, alpha=0.7, linewidths=0, zorder=2)
            fit = fit_buffering(g[TOTAL], g[DILUTE])
            if np.isfinite(fit["c_sat"]):
                x = np.log10(np.linspace(g[TOTAL].min(), g[TOTAL].max(), 300))
                k = np.log10(fit["c_sat"])
                yseg = np.where(x < k, fit["a"] + x, fit["a"] + k + fit["beta"] * (x - k))
                yline = fit["a_line"] + fit["beta_line"] * x
                ax.plot(10 ** x, 10 ** yline, color="#555555", linewidth=1.2, linestyle="-.",
                        zorder=3, label=f"line: slope {fit['beta_line']:.2f}")
                ax.plot(10 ** x, 10 ** yseg, color=c, linewidth=2.4, zorder=4,
                        label=f"segmented: β = {fit['beta']:.2f}")
                row = table[(table["condition"] == cond) & (table["method"] == "buffering")].iloc[0]
                if row["verdict"] == "ok":
                    ax.axvline(fit["c_sat"], color=c, linestyle=":", linewidth=1.4)
                    ax.text(fit["c_sat"], lims[0] * 1.15, f" c_sat ≈ {fit['c_sat']:.3g}",
                            color=c, fontsize=8.5, fontweight="bold", va="bottom")
            ax.set_xscale("log")
            ax.set_yscale("log")
            for axis in (ax.xaxis, ax.yaxis):
                _fmt_log(axis)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title(f"{cond}  (n = {len(g)})")
            ax.set_xlabel("total (nuclear intensity)")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            sns.despine(ax=ax)
        axes[0, 0].set_ylabel("dilute (nucleoplasm intensity)")
        fig.suptitle("Dilute-phase buffering", fontsize=13, fontweight="bold")
        fig.tight_layout()
    return fig


def plot_lever(df, palette, table):
    """Fraction of protein in foci vs total, with the lever-rule fit."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        for cond, g in df.groupby("condition", observed=True):
            c = palette[cond]
            ax.scatter(g[TOTAL], g["fraction_signal_in_foci"], s=20, color=c, alpha=0.7,
                       linewidths=0, label=str(cond))
            fit = fit_lever(g[TOTAL], g["fraction_signal_in_foci"])
            row = table[(table["condition"] == cond) & (table["method"] == "lever")].iloc[0]
            if np.isfinite(fit["c_sat"]):
                x = np.geomspace(max(fit["c_sat"], g[TOTAL].min() * 0.5), g[TOTAL].max(), 300)
                ok = row["verdict"].startswith(("ok", "below"))
                ax.plot(x, _lever(x, fit["A"], fit["c_sat"]), color=c, linewidth=2.2 if ok else 1.2,
                        linestyle="-" if ok else "--")
                if row["verdict"] == "ok":
                    ax.axvline(fit["c_sat"], color=c, linestyle=":", linewidth=1.3)
        _log_axis(ax)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("total (nuclear intensity)")
        ax.set_ylabel("fraction of nuclear signal in foci")
        ax.set_title("Lever rule: 1 − c_sat / total", pad=10)
        ax.legend(frameon=False)
        sns.despine(ax=ax)
        fig.tight_layout()
    return fig


def plot_estimates(table, palette):
    """Forest plot: c_sat per method (panel) and condition, with CI.
    Unreliable estimates (verdict not ok) are drawn hollow."""
    methods = [m for m in ("onset", "buffering", "lever") if m in set(table["method"])]
    labels = {"onset": "Onset (50 % of cells with foci)", "buffering": "Dilute-phase buffering",
              "lever": "Lever rule"}
    conds = list(dict.fromkeys(table["condition"]))
    shown = ~table["verdict"].str.contains("not identifiable|not estimable")
    finite = table.loc[shown, ["estimate", "ci_low", "ci_high"]].to_numpy(float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    with plt.rc_context({**STYLE, "axes.grid.axis": "x"}):
        fig, axes = plt.subplots(len(methods), 1, sharex=True, squeeze=False,
                                 figsize=(7.4, 0.42 * len(conds) * len(methods) + 0.55 * len(methods) + 0.9))
        for ax, m in zip(axes[:, 0], methods):
            for k, cond in enumerate(conds):
                r = table[(table["condition"] == cond) & (table["method"] == m)]
                y = len(conds) - 1 - k
                if r.empty:
                    continue
                r = r.iloc[0]
                ok = r["verdict"] == "ok"
                c = palette[cond]
                if "not identifiable" in r["verdict"] or "not estimable" in r["verdict"]:
                    ax.text(0.02, y, "not identifiable" if "identifiable" in r["verdict"] else "not estimable",
                            transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                            style="italic", color="#999999")
                    continue
                if np.isfinite(r["ci_low"]) and r["ci_low"] > 0:
                    ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=c, linewidth=3.2,
                            alpha=0.65 if ok else 0.25, solid_capstyle="round")
                if np.isfinite(r["estimate"]) and r["estimate"] > 0:
                    ax.scatter(r["estimate"], y, s=95, color=c if ok else "white",
                               edgecolor="black" if ok else c, linewidths=1.5, zorder=3)
                else:
                    ax.text(0.02, y, "not estimable", transform=ax.get_yaxis_transform(),
                            va="center", fontsize=8, style="italic", color="#999999")
            ax.set_yticks(range(len(conds)))
            ax.set_yticklabels(conds[::-1])
            ax.set_ylim(-0.6, len(conds) - 0.4)
            ax.tick_params(axis="y", length=0)
            ax.set_title(labels[m], loc="left", fontsize=10, pad=4)
            sns.despine(ax=ax, left=True)
        ax = axes[-1, 0]
        ax.set_xscale("log")
        if len(finite):
            ax.set_xlim(finite.min() / 1.25, finite.max() * 1.25)
        _fmt_log(ax.xaxis)
        ax.set_xlabel("c_sat (nuclear intensity units)")
        fig.suptitle("c_sat estimates  ·  hollow = not reliable (see verdict)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
    return fig
