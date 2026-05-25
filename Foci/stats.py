import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
#from scipy.stats import mannwhitneyu

def compare_to_reference(
    dfs,
    col,
    reference_idx=0,
    equal_var=True,
    alpha=0.05
):
    """
    Compare all dataframes to a reference dataframe using t-test.

    Parameters
    ----------
    dfs : list of pandas DataFrame
    col : str
        Column for comparison
    reference_idx : int
        Index of reference dataframe
    equal_var : bool
        Student t-test if True, Welch t-test if False
    alpha : float
        Significance threshold

    Returns
    -------
    pandas.DataFrame
    """

    reference = dfs[reference_idx]

    results = []

    for i, group in enumerate(dfs):

        if i == reference_idx:
            continue

        x = reference[col].dropna()
        y = group[col].dropna()

        t_stat, p_value = ttest_ind(
            x,
            y,
            equal_var=equal_var
        )

        results.append({
            "comparison": f"dfs[{reference_idx}] vs dfs[{i}]",
            "n_reference": len(x),
            "n_group": len(y),
            "mean_reference": x.mean(),
            "mean_group": y.mean(),
            "std_reference": x.std(),
            "std_group": y.std(),
            "t_stat": t_stat,
            "p_value": p_value,
            "significant": p_value < alpha
        })

    result_df = pd.DataFrame(results)

    result_df["p_value"] = result_df["p_value"].map(lambda x: f"{x:.3e}")
    result_df["t_stat"] = result_df["t_stat"].map(lambda x: f"{x:.3f}")

    return result_df

def make_stat_pairs_from_results(stats_df, labels, reference_name="Control"):
    stat_pairs = []

    x1 = labels.index(reference_name) + 1  # boxplot positions start from 1

    for _, row in stats_df.iterrows():
        comparison = row["comparison"]

        group_name = comparison.split(" vs ")[1]
        x2 = labels.index(group_name) + 1

        p_value = float(row["p_value"])

        stat_pairs.append((x1, x2, p_value))

    return stat_pairs

def p_to_stars(p):
    if p <= 0.0001:
        return "****"
    elif p <= 0.001:
        return "***"
    elif p <= 0.01:
        return "**"
    elif p <= 0.05:
        return "*"
    else:
        return "ns"

def add_stat_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color="black", linewidth=1)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)

def compare_to_reference(
    dfs,
    data,
    col,
    reference_idx=0,
    equal_var=True,
    alpha=0.05
):
    """
    Compare all dataframes to a reference dataframe using t-test.

    Parameters
    ----------
    dfs : list of pandas DataFrame
    col : str
        Column for comparison
    reference_idx : int
        Index of reference dataframe
    equal_var : bool
        Student t-test if True, Welch t-test if False
    alpha : float
        Significance threshold

    Returns
    -------
    pandas.DataFrame
    """

    reference = dfs[reference_idx]
    reference_name = data[reference_idx]["name"]

    results = []

    for i, group in enumerate(dfs):
        if i == reference_idx:
            continue
        
        group_name = data[i]["name"]

        x = reference[col].dropna()
        y = group[col].dropna()

        t_stat, p_value = ttest_ind(
            x,
            y,
            equal_var=equal_var
        )

        results.append({
            "comparison": f"{reference_name} vs {group_name}",
            "variable_name": col,
            "n_reference": len(x),
            "n_group": len(y),
            "mean_reference": x.mean(),
            "mean_group": y.mean(),
            "std_reference": x.std(),
            "std_group": y.std(),
            "t_stat": t_stat,
            "p_value": p_value,
            "significant": p_value < alpha
        })

    result_df = pd.DataFrame(results)

    result_df["p_value"] = result_df["p_value"].map(lambda x: f"{x:.3e}")
    result_df["t_stat"] = result_df["t_stat"].map(lambda x: f"{x:.3f}")

    return result_df

def beautiful_boxplot(
    df_list,
    labels,
    ylabel=None,
    xlabel=None,
    title=None,
    log_scale=False,
    colors=None,         # list of colors per box (optional)
    dot_size=20,
    jitter=0.06,
    figsize=(4.8, 4.2),
    dpi=300,
    show=False,
    stats_df=None,
    reference_name="Control",
    stat_pairs=None,
    save=True,
    path_to_save = None    
):
    """
    df_list : list of pandas Series (or 1D arrays)
    labels  : list of group names (same length as df_list)
    colors  : list of colors for each box (same length as df_list) or None
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Clean data
    data = [np.asarray(d.dropna()) for d in df_list]

    # Boxplot
    bp = ax.boxplot(
        data,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )

    # Colors per box (or default)
    if colors is None:
        colors = ["lightgray"] * len(data)

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)

    # Overlay jittered dots
    for i, y in enumerate(data, start=1):
        x = np.random.normal(loc=i, scale=jitter, size=len(y))
        ax.scatter(x, y, alpha=0.65, s=dot_size, linewidths=0)

    # Styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=4, width=1, labelsize=9)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, pad=10)

    if log_scale:
        ax.set_yscale("log")

    # Create stat_pairs automatically from stats_df
    if stat_pairs is None and stats_df is not None:
        stat_pairs = make_stat_pairs_from_results(
            stats_df=stats_df,
            labels=labels,
            reference_name=reference_name
        )
    
    # Add significance brackets
    if stat_pairs:
        y_max = max([np.nanmax(y) for y in data]) # find the largest value
        y_min = min([np.nanmin(y) for y in data]) # find the minimal value
        y_range = y_max - y_min

        if y_range == 0:
            y_range = y_max * 0.1 if y_max != 0 else 1

        for idx, (x1, x2, p_value) in enumerate(stat_pairs):
            y = y_max + y_range * (0.08 + idx * 0.12)
            h = y_range * 0.04
            stars = p_to_stars(p_value)

            add_stat_bracket(ax, x1, x2, y, h, stars)

        ax.set_ylim(top=y_max + y_range * (0.2 + len(stat_pairs) * 0.12))

    fig.tight_layout()

    if show:
        plt.show()
    
    if save:
        fig.savefig(
        path_to_save,
        dpi=dpi,              # resolution
        bbox_inches="tight",  # removes extra white margins
        transparent=False     # transparent background if True
        )


    return fig, ax

def main():
    # Paths to tables with data
    reference_name = 'Control'  
    data = [
        {'name': reference_name, 'path': "/mnt/c/users/elopatukhin/Desktop/all_WT/foci_aggregation.csv"},
        {'name': 'PDS, 5 uM', 'path': "/mnt/c/users/elopatukhin/Desktop/all_PDS_5uM/foci_aggregation.csv"},
        {'name': 'PDS, 20 uM', 'path': "/mnt/c/users/elopatukhin/Desktop/all_PDS_20uM/foci_aggregation.csv"}
    ]

    # Index of Control sample in the list of dictionaries 
    index_control = next(
        i for i, item in enumerate(data)
        if item["name"] == "Control"
    )
    
    # Path to save plot
    output_dir = "/mnt/c/users/elopatukhin/Desktop/all_WT"
    
    # Arguments for boxplot
    args = [{
            'var': "n_foci",
            'ylabel':'Number of foci',
            'title':''
            },
            {
            'var': "sigma_nm_mean",
            'ylabel':'Sigma, nm',
            'title':''
            },
            {
            'var': "foci_MFI_mean",
            'ylabel':'Mean Fluorescent intensity of foci',
            'title':''
            }]

    # Load data
    dfs = []
    for item in data:
        p = item["path"]
        df = pd.read_csv(p)
        dfs.append(df)

    N_files = len(dfs)
    print(f"Loaded {N_files} dataframes.")

    # Loop to make statistics and plot
    for arg in args:
        # Take desired column 'var' of all dataframes.
        selected = [df[arg['var']] for df in dfs]

        # Calculate statistics
        stats = compare_to_reference(dfs = dfs,
                                    data = data,
                                    col = arg['var'],
                                    reference_idx=index_control,
                                    equal_var=True,
                                    alpha=0.05
                                    )
        
        # Path to save statistics
        stats_name = f"statistics_{arg['var']}.csv"
        full_path_to_stats = os.path.join(output_dir, stats_name)

        # Save statistics
        stats.to_csv(full_path_to_stats, index=False)
        print(f"Statistics for {arg['var']} is saved in the directory: {output_dir}")

        # Path to save plot
        plot_name = f"plot_{arg['var']}.png"
        full_path_to_plot = os.path.join(output_dir, plot_name)

        # Make boxplot and save
        beautiful_boxplot(
        df_list = selected,
        labels = [item["name"] for item in data],
        ylabel=arg['ylabel'],
        xlabel=None,
        title=arg['title'],
        log_scale=False,
        colors=None,      
        dot_size=20,
        jitter=0.06,
        figsize=(4.8, 4.2),
        dpi=300,
        show=False,
        stats_df=stats,
        reference_name=reference_name,
        save=True,
        path_to_save = full_path_to_plot
        )

        print(f"Plot for {arg['var']} is saved in the directory: {output_dir}")
        print("\n")



if __name__ == "__main__":
    main()