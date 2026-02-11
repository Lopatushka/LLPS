from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import re

def base_name_from_csv(filename: str) -> str:
    """
    Turn 'sample_0262-0212.csv' -> 'sample'
    Turn 'sample.csv' -> 'sample'
    """
    # remove extension
    name = re.sub(r"\.csv$", "", filename)
    # remove trailing _0262-0212 (or similar) if present
    name = re.sub(r"_\d+-\d+$", "", name)
    return name

def aggregate_data(dir1, dir2):
    # normalize paths (strip accidental spaces)
    dir_path1 = Path(str(dir1).strip())
    dir_path2 = Path(str(dir2).strip())

    if not dir_path1.exists():
        raise FileNotFoundError(f"dir1 not found: {dir_path1}")
    if not dir_path2.exists():
        raise FileNotFoundError(f"dir2 not found: {dir_path2}")
    
    # ---- NUCLEI TABLE (path1) ----
    files1 = sorted(dir_path1.glob("*.csv"))
    if not files1:
        raise FileNotFoundError(f"No CSV files found in dir1: {dir_path1}")
    
    dfs1 = []
    for f in files1:
        key = base_name_from_csv(f.name)
        key = key[:-4]
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()  # remove hidden spaces in headers

        # ensure expected columns exist
        if "Area" not in df.columns or "Mean" not in df.columns:
            raise KeyError(
                f"In nuclei file {f.name} expected columns 'Area' and 'Mean'. "
                f"Found: {list(df.columns)}"
            )
        
        df["File_name"] = key
        df = df.rename(columns={"Area": "Nucleus_area", "Mean": "Nucleus_MFI"})
        df = df[["File_name", "Nucleus_area", "Nucleus_MFI"]]
        dfs1.append(df)

    final = pd.concat(dfs1, ignore_index=True)

    # ---- FOCI SUMMARY (path2) ----
    files2 = sorted(dir_path2.glob("*.csv"))
    if not files2:
        raise FileNotFoundError(f"No CSV files found in dir2: {dir_path2}")
    
    foci_rows = []
    for f in files2:
        key = base_name_from_csv(f.name)
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        # count rows + mean intensity
        foci_rows.append({
        "File_name": key,
        "Foci_number": int(df.shape[0]),
        "Foci_MFI": float(df["intensity [photon]"].mean()) if "intensity [photon]" in df.columns and df.shape[0] > 0 else pd.NA
    })

    foci_summary = pd.DataFrame(foci_rows)

    # ---- MERGE ----
    final["File_name"] = final["File_name"].astype(str).str.strip()
    foci_summary["File_name"] = foci_summary["File_name"].astype(str).str.strip()

    merged  = final.merge(foci_summary, on="File_name", how="left")
    merged["Foci_MFI"] = pd.to_numeric(merged["Foci_MFI"], errors="coerce")

    return merged

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

def main(path1, path2, out_dir):
    path1 = str(path1).strip()
    path2 = str(path2).strip()
    out_dir = Path(str(out_dir).strip())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make final table
    results = aggregate_data(path1, path2)

    # Spearman correlation
    corr = sprearman_correlation(results)

    # Results export
    results.to_csv(out_dir / "results.csv", index=False)
    corr.to_csv(out_dir / "spearman_pairs.csv", index=False)

    print(f"Saved: {out_dir / 'results.csv'}")
    print(f"Saved: {out_dir / 'spearman_pairs.csv'}")

 

if __name__ == "__main__":
    path1 =  "/mnt/c/Users/Elena/Desktop/Data_processing/020226_U2OS_fixed_WT" # path to the folder containing the csv files with nucleus area and MFI
    path2 = "/mnt/c/Users/Elena/Desktop/Data_processing/020226_U2OS_fixed_WT/res8" # path to the folder containing the csv files with foci number and MFI
    path3 = path2
    main(path1, path2, path3)