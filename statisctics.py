from ij import IJ, WindowManager
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import re

def base_name_from_csv(filename: str) -> str:
    # remove extension
    name = re.sub(r"\.csv$", "", filename)
    # remove trailing _0262-0212 (or similar) if present
    name = re.sub(r"_\d+-\d+$", "", name)
    return name

def aggregate_data(dir1, dir2):
    # ---- NUCLEI TABLE (path1) ----
    #dir_path1 = Path(path1)
    files1 = sorted(dir1.glob("*.csv"))
    dfs1 = []
    for f in files1:
        key = base_name_from_csv(f.name)
        key = key[:-4]
        df = pd.read_csv(f)
        df["File_name"] = key
        dfs1.append(df)

    final = pd.concat(dfs1, ignore_index=True)
    final = final.rename(columns={"Area": "Nucleus_area", "Mean": "Nucleus_MFI"})
    final = final[["File_name", "Nucleus_area", "Nucleus_MFI"]]

    # ---- FOCI SUMMARY (path2) ----
    #dir_path2 = Path(path2)
    files2 = sorted(dir2.glob("*.csv"))
    foci_rows = []
    for f in files2:
        key = base_name_from_csv(f.name)
        df = pd.read_csv(f)
        # count rows + mean intensity
        foci_rows.append({
            "File_name": key,
            "Foci_number": int(df.shape[0]),
            "Foci_MFI": float(df["intensity [photon]"].mean()) if "intensity [photon]" in df.columns else pd.NA
        })

    foci_summary = pd.DataFrame(foci_rows)

    # ---- MERGE ----
    final = final.merge(foci_summary, on="File_name", how="left")
    final["Foci_MFI"] = pd.to_numeric(final["Foci_MFI"], errors="coerce")

    return final

def sprearman_correlation(df):
    cols = df.select_dtypes(include="number").columns
    pairs = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            x, y = df[c1], df[c2]
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n > 2:
                r, p = spearmanr(x[mask], y[mask])
                pairs.append({"var1": c1, "var2": c2, "n": n, "spearman_r": r, "p_value": p})

    pairs_df = pd.DataFrame(pairs)
    return pairs_df


def main():
    # Ask user about directory with .csv files
    path1 = IJ.getDirectory("Choose a directory with .csv files with nucleus Area and MFI")
    path2 = IJ.getDirectory("Choose a directory with .csv files containing the number of foci and their MFI")
    if path1 is None or path2 is None:
        IJ.error("No directory selected. Exiting.")
        raise SystemExit

    try:
        # Make final table
        IJ.log("Processing files...")
        results = aggregate_data(path1, path2)
        IJ.log("Processing completed successfully.")

        # Spearman correlation
        corr = sprearman_correlation(results)

        # Results export
        path3 = IJ.getDirectory("Choose a directory to save the results file")
        results.to_csv(path3 + "/results.csv", index=False)
        corr.to_csv(path3 + "/spearman_pairs.csv", index=False)

    except Exception as e:
        IJ.error(f"Error processing files: {e}")
        raise SystemExit

if __name__ == "__main__":
    main()