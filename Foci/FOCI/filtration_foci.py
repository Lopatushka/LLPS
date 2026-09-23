import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# For Ayriscan detector filration conditions are the following:
#
# Sigma, nm - [100, 1000]
# MFI > 100
# MFI/SD of FI > 5
# If to circles are overllaped > 30%, keep only the biggest

def plot_histogram(df, column, bins=50,
                   xlabel=None,
                   title=None,
                   figsize=(4, 3),
                   dpi=300,
                   save_image = True,
                   save_path=None):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.hist(
        df[column].dropna(),
        bins=bins,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=12)

    #ax.axvline(threshold, linestyle="--", linewidth=2)

    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    #plt.show()

    # --- Save if path provided ---
    if save_image:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    
def df_filtration(df_path, plot = True):
    # Process the filename
    path = Path(df_path)
    file_name = path.stem
    
    # Load dataframe
    df = pd.read_csv(path)
        
    # --- Perform filrations ---
    # Sigma filtation
    df = df[(df['sigma_nm'] > 100) & (df['sigma_nm'] < 1000)]
    
    # MFI filtration
    df = df[df['foci_MFI'] > 100]
    
    # S.d. filtration
    df[df['foci_MFI'] / df['foci_SD'] > 8]
    
    if plot:
        # Plot and save histograms of original images
        plot_histogram(df, column = "sigma_nm", bins=50,
                    xlabel="Sigma, nm",
                    title=file_name,
                    figsize=(4, 3),
                    dpi=300,
                    save_image = True,
                    save_path = path.with_name(file_name + "_sigma_hist.png")
                    )
    
        plot_histogram(df, column = "foci_MFI", bins=50,
                    xlabel="MFI of foci",
                    title=file_name,
                    figsize=(4, 3),
                    dpi=300,
                    save_image = True,
                    save_path = path.with_name(file_name + "_foci_MFI_hist.png")
                    )
    
        plot_histogram(df, column = "foci_SD", bins=50,
                    xlabel="S.d. of fluorescent intencity of foci",
                    title=file_name,
                    figsize=(4, 3),
                    dpi=300,
                    save_image = True,
                    save_path = path.with_name(file_name + "_foci_sd_hist.png")
                    )

# -----------------
# MAIN FUNCTION
# -----------------
def main():
    pass

if __name__ == "__main__":
    main()