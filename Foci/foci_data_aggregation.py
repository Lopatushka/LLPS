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

def foci_data():
    return

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