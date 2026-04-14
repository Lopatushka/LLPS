import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

def check_dir_exists(path, ensure_dir = True):
    path = Path(path)

    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}!")
    else:
        if ensure_dir:
            path.mkdir(parents=True, exist_ok=True)
            print(f"Directory {path} is created.")
        else:
            raise NotADirectoryError(f"There is no such directory: {path}!")

    return path

def check_file_exists(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if not path.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")

    return path


def manage_dir(path, verbose=True):
    path = Path(path)

    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {path}")
        if verbose:
            print(f"Directory already exists: {path}")
    else:
        path.mkdir(parents=True)
        if verbose:
            print(f"Created directory: {path}")

    return path

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
    show=True
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

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax

def mannwhitneyu_stats(data,
                       column_name,
                       reference_index = 0,
                       save = True,
                       output_folder = None):
    """
    Perform Mann–Whitney U tests comparing a reference group to other groups.

    Parameters:
    - data: list of DataFrames
    - column_name: column to analyze
    - reference_index: index of reference group
    - save: whether to save results as CSV
    - output_folder: folder to save results (Path or str)

    Returns:
    - results_df: DataFrame with test statistics
    """
        
    # Extract the reference data
    reference = data[reference_index]

    # Create empty list
    results = []

    # Loop through the list of dataframes
    for i in range(1, len(data)):
        group = data[i]

        # Drop NaNs
        x = reference[column_name].dropna()
        y = group[column_name].dropna()

        stat, p_value = mannwhitneyu(x, y)

        results.append({
            "comparison": f"WT vs MGS{i}",
            "stat": stat,
            "p_value": p_value,
            'significant' : p_value < 0.06
        })

    # Convert list to dataframe
    results_df = pd.DataFrame(results)

    # Save data
    if save:
        results_df.to_csv(output_folder / f"{column_name}_MW_stats.csv", index=False)
        print(f"The results are saved into the folder: {output_folder}")

    return results_df