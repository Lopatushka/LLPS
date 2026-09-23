import os
import pandas as pd
import numpy as np
from PIL import Image
from skimage.draw import disk
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

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

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"The directrory {path} is created.")
        
def filename(path):
    """
    Return filename without extenstion
    """
    return os.path.splitext(os.path.basename(path))[0]

def draw_foci(image, df, showplot = True, save_image = True, save_path = ""):
    # Load image
    #image = Image.open(image_path)
    arr = np.array(image) # convert image to numpy matrix

    # Plot image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr, cmap="gray")

    # Draw red circles
    for _, row in df.iterrows():
        x = row["x_pixel"]
        y = row["y_pixel"]
        r = row["sigma_pixel"]

        circle = Circle(
            (x, y),
            r,
            fill=False,
            edgecolor="red",
            linewidth=1
        )

        ax.add_patch(circle)

    # Match image coordinates
    ax.set_xlim(0, arr.shape[1])
    ax.set_ylim(arr.shape[0], 0)

    # Show image
    if showplot:
        plt.show(fig)

    # Save image
    if save_image: 
        plt.savefig(save_path,
            dpi=300,
            bbox_inches="tight"
        )

        # Do not display image
        plt.close(fig)

def foci_one_image(image, df, px_size_nm, plot = True, save_path = ""):
    arr = np.array(image) # convert image to numpy matrix
    H, W = arr.shape # get number of pixels (512*512 for 16-bit image)

    # Storage lists
    x_list = []
    y_list = []
    sigma_list = []
    mean_list = []
    sd_list = []
    
    # Prepare dataframe
    df.columns = df.columns.str.strip()  # remove hidden spaces in headers
    # Rename columns
    df = df.rename(columns={"x [nm]": "x_nm",
                            "y [nm]": "y_nm",
                            "sigma [nm]": "sigma_nm",
                            "intensity [photon]": "intensity_photon"})
    
    # Iteration through the ThunderSTORM dataframe
    for _, row in df.iterrows():
        x_px = int(row["x_nm"] / px_size_nm)
        y_px = int(row["y_nm"] / px_size_nm)
        r_px = max(1, int(row["sigma_nm"] / px_size_nm)) # minimal possible value for radius is 1 pixel!

        # Build circular mask (clipped automatically)
        rr, cc = disk((y_px, x_px), r_px, shape=(H, W))
        mask = np.zeros((H, W), dtype=bool)
        mask[rr, cc] = True

        n_pixels_mask = np.sum(mask)

        # Compute mean intensity
        if n_pixels_mask > 0:
            mean_intensity = arr[mask].mean()
            sd_intensity = arr[mask].std()
        else:
            mean_intensity = np.nan
            sd_intensity = np.nan

        # Add values to the corresponding lists
        x_list.append(x_px)
        y_list.append(y_px)
        sigma_list.append(r_px)
        mean_list.append(mean_intensity)
        sd_list.append(sd_intensity)
        
    # Return modified copy
    df_out = df.copy()
    df_out["x_pixel"] = x_list
    df_out["y_pixel"] = y_list
    df_out["sigma_pixel"] = sigma_list
    df_out["foci_MFI"] = mean_list
    df_out["foci_SD"] = sd_list
    
    # Make a plot
    if plot:
        draw_foci(image = image,
                  df = df_out,
                  showplot = False,
                  save_image = True,
                  save_path = save_path)

    return df_out

#---------------------
# MAIN FUNCTION
#---------------------
def main():
    # Ask user about the path to the directory with images and foci.csv files
    dir_images = check_directory(input("Enter pathway to the directory with the images: "))
    dir_foci = check_directory(input("Enter pathway to the directory with ThunderSTORM output in .csv format): "))
    
    px = float(input("Enter the pixel size in nm [default value is 35.3]: ") or 35.3)
    
    while True:
        answer = input("Save results in the same folder as foci? (Y/N): ").strip().upper()
        if answer == "Y":
            output_dir = dir_foci
            break
        elif answer == "N":
            output_dir = check_directory(input("Enter output folder path: ").strip())
            break
        else:
            print("Please enter Y or N.")
    
    # -------------------------
    # --- Process foci data ---
    # -------------------------
    
    # List of paths to the images
    paths_images = [
    os.path.join(dir_images, f)
    for f in os.listdir(dir_images)
    if os.path.isfile(os.path.join(dir_images, f))
    and f.lower().endswith(".tif") and "_roi_".lower() in f.lower()
    ]
    
    print(f"Number of founded images is {len(paths_images)}")
    
    # List of paths to the foci.csv
    paths_foci_csv = [
        os.path.join(dir_foci, f)
        for f in os.listdir(dir_foci)
        if os.path.isfile(os.path.join(dir_foci, f))
        and f.lower().endswith(".csv")   
        ]
    
    print(f"Number of founded .csv files is {len(paths_foci_csv)}")
    
    # Create dictionaries
    img_by_key = {filename(image_name).replace(" ", "_"): image_name for image_name in paths_images} # dictionary {image name w/o ext: image path}
        
    csv_by_key = {filename(csv_name)[:-5].replace(" ", "_"): csv_name for csv_name in paths_foci_csv} # dictionary {csv file name w/o ext: image path}
        
    combined = {k: (img_by_key[k], csv_by_key[k]) for k in img_by_key} # dictionary {file_name: (path_to_image, path_to_foci_csv)}
    
    n_images = len(combined)
    print(f"Founded {n_images} pairs of image.tif : foci.csv files.")
    
    # Iteration through combined dictioanry
    for name, (img_path, csv_path) in combined.items():
        try:
            # Open image
            image = Image.open(img_path)
            
            # Read foci.csv file
            df = pd.read_csv(csv_path)
            
            #plot_path = os.path.join(output_dir, f"{name}_foci_mapped.png")
            result = foci_one_image(image = image,
                                    df = df,
                                    px_size_nm = px,
                                    plot = False,
                                    save_path = "")
            
            # Save the result to a new .csv file
            result_csv_path = os.path.join(output_dir, f"{name}_foci_mapped.csv")
            result.to_csv(result_csv_path, index=False)
            
            print(f"Sucessfully processed image {name}.")
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue
        
if __name__ == "__main__":
    main()