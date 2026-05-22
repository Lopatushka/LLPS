import os
import pandas as pd

def check_directory(path):
    if not isinstance(path, str):
        raise TypeError("Path must be a string.")
    
    path = path.strip()
    if path == "":
        raise ValueError("Path is empty.")
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Directory does not exist:\n{}".format(path)
        )

    if not os.path.isdir(path):
        raise NotADirectoryError(
            "Path is not a directory:\n{}".format(path)
        )

    return os.path.abspath(path)

def filename(path):
    """
    Return filename without extenstion
    """
    return os.path.splitext(os.path.basename(path))[0]

def foci_data(dir, output_dir):
    paths_csv_files = [
    os.path.join(dir, f)
    for f in os.listdir(dir)
    if os.path.isfile(os.path.join(dir, f))
    and f.lower().endswith(".csv")
    ]

    # Number of .csv files and check
    n = len(paths_csv_files)
    if n == 0:
        raise FileNotFoundError(f"No CSV files found in the directory: {dir}")
    print(f"Founded {n} .csv files")

    # Create a list of dictionaries
    rows = []
    
    # Create list of dataframes
    for f in paths_csv_files:
        name = filename(f)
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()  # remove hidden spaces in headers

        # Number of foci
        n_foci = df.shape[0]

        # Calculate mean and sd values
        sigma_nm_mean = df["sigma_nm"].mean()
        sigma_nm_sd = df["sigma_nm"].sd()

        intensity_photon_mean = df["intensity_photon"].mean()
        intensity_photon_sd = df["intensity_photon"].sd()

        sigma_pixel_mean = df["sigma_pixel"].mean()
        sigma_pixel_sd = df["sigma_pixel"].sd()

        foci_MFI_mean = df["foci_MFI"].mean()
        foci_MFI_sd = df["foci_MFI"].sd()

        rows.append({
            "filname": name,
            "n_foci": n_foci,
            "sigma_nm_mean": sigma_nm_mean,
            "sigma_nm_sd": sigma_nm_sd,
            "intensity_photon_mean": intensity_photon_mean,
            "intensity_photon_sd": intensity_photon_sd,
            "sigma_pixel_mean": sigma_pixel_mean,
            "sigma_pixel_sd": sigma_pixel_sd,
            "foci_MFI_mean": foci_MFI_mean,
            "foci_MFI_sd": foci_MFI_sd
        })

    final = pd.DataFrame(rows)

    # Save final file
    path_to_save = os.path.join(output_dir, "foci_aggregation.csv")
    final.to_csv(path_to_save, index=False)


def main():
    # Ask about paths with data and output directory to save results
    # Example of path: /mnt/c/users/elopatukhin/Desktop/Miscroscopy/160226_U2OS_fixed/MP_WT_0.3
    foci_dir = check_directory(input("Enter pathway to the directory with the information about foci: "))
    while True:
        answer = input("Save results in the same folder as foci? (Y/N): ").strip().upper()
        if answer == "Y":
            output_dir = foci_dir
            break
        elif answer == "N":
            output_dir = check_directory(input("Enter output folder path: ").strip())
            break
        else:
            print("Please enter Y or N.")

    # --- Processed nuclei info (Area, Mean) ---
    foci_data(foci_dir, output_dir)

 
if __name__ == "__main__":
    main()