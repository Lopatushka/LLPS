import os
import re
import pandas as pd
import numpy as np
from PIL import Image
from skimage.draw import disk
from scipy.ndimage import gaussian_filter, map_coordinates
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

# ThunderSTORM puts the coordinate origin at the top-left CORNER of the image,
# so the centre of pixel (0, 0) is at (0.5, 0.5) px. NumPy/skimage index pixel
# centres at integers, hence: pixel_index = coordinate_nm / pixel_size_nm - 0.5
THUNDERSTORM_ORIGIN_SHIFT_PX = -0.5

# Warn if the pixel size that best aligns localizations with image intensity
# differs from the pixel size used by more than this fraction.
PIXEL_SIZE_TOLERANCE = 0.005

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

def tiff_pixel_size_nm(image):
    """
    Pixel size (nm) from ImageJ TIFF tags. ImageJ stores XResolution as
    pixels per unit, with the unit in the ImageDescription (e.g. unit=micron).
    """
    x_res = image.tag_v2.get(282)  # XResolution
    if x_res is None or float(x_res) == 0:
        return None
    desc = str(image.tag_v2.get(270) or "")
    m = re.search(r"unit=(\S+)", desc)
    unit = m.group(1).lower() if m else "micron"
    factor = {"nm": 1.0, "micron": 1000.0, "um": 1000.0, "\\u00b5m": 1000.0, "µm": 1000.0}.get(unit)
    if factor is None:
        return None
    return factor / float(x_res)

def protocol_pixel_size_nm(csv_path):
    """
    Pixel size (nm) ThunderSTORM actually used, read from the protocol file
    written next to the CSV (<csv name>-protocol.txt). None if unavailable.
    """
    protocol = os.path.splitext(csv_path)[0] + "-protocol.txt"
    if not os.path.isfile(protocol):
        return None
    with open(protocol, encoding="utf-8", errors="ignore") as f:
        m = re.search(r'"pixelSize"\s*:\s*([0-9.eE+-]+)', f.read())
    return float(m.group(1)) if m else None

def estimate_pixel_size_nm(arr, x_nm, y_nm, guess_nm, rel_range=0.05, step_nm=0.05):
    """
    QC: find the pixel size that maximizes the mean (smoothed) image intensity
    at the localizations. A wrong pixel size misplaces foci progressively
    with distance from the origin, which lowers this score.
    Returns None if there are too few foci for a reliable estimate.
    """
    if len(x_nm) < 30:
        return None
    sm = gaussian_filter(np.asarray(arr, dtype=float), 1)
    candidates = np.arange(guess_nm * (1 - rel_range), guess_nm * (1 + rel_range), step_nm)
    scores = [
        map_coordinates(
            sm,
            [y_nm / p + THUNDERSTORM_ORIGIN_SHIFT_PX, x_nm / p + THUNDERSTORM_ORIGIN_SHIFT_PX],
            order=1
        ).mean()
        for p in candidates
    ]
    return float(candidates[int(np.argmax(scores))])

def draw_foci(image, df, showplot=True, save_image=True, save_path=""):
    # Convert image to numpy array
    arr = np.array(image)

    # Plot image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
    arr,
    cmap="gray",
    origin="upper",
    interpolation="nearest"
    )

    # Draw red dots at foci centers
    ax.scatter(
        df["x_pixel"],
        df["y_pixel"],
        s=4,               # Marker size
        c="red",
        marker="+",
        linewidths=0.25
    )

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
            linewidth=0.45
        )

        ax.add_patch(circle)

    # Match image coordinates
    ax.set_xlim(-0.5, arr.shape[1] - 0.5)
    ax.set_ylim(arr.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")

    # Save image before displaying
    if save_image:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    # Show image
    if showplot:
        plt.show()

    plt.close(fig)

def foci_one_image(image, df, px_size_nm, plot=True, save_path="", name=""):
    """
    Convert ThunderSTORM localizations to pixel coordinates and measure the
    mean/SD intensity of each focus inside a circle of radius sigma.

    px_size_nm must be the pixel size ThunderSTORM used for this image.
    """
    arr = np.array(image)
    H, W = arr.shape

    # Prepare dataframe
    df = df.copy()
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "x [nm]": "x_nm",
        "y [nm]": "y_nm",
        "sigma [nm]": "sigma_nm",
        "intensity [photon]": "intensity_photon"
    })

    # QC: does the pixel size align localizations with the image?
    best = estimate_pixel_size_nm(arr, df["x_nm"].to_numpy(), df["y_nm"].to_numpy(), px_size_nm)
    if best is not None and abs(best - px_size_nm) / px_size_nm > PIXEL_SIZE_TOLERANCE:
        print(
            f"  WARNING {name}: localizations align best with pixel size "
            f"{best:.2f} nm, but {px_size_nm:.2f} nm is used. Foci positions and "
            f"MFI may be wrong; check the ThunderSTORM camera setup."
        )

    # Pixel coordinates (pixel-centre convention, see THUNDERSTORM_ORIGIN_SHIFT_PX)
    x_px = df["x_nm"].to_numpy() / px_size_nm + THUNDERSTORM_ORIGIN_SHIFT_PX
    y_px = df["y_nm"].to_numpy() / px_size_nm + THUNDERSTORM_ORIGIN_SHIFT_PX
    r_px = df["sigma_nm"].to_numpy() / px_size_nm

    mean_list = []
    sd_list = []
    for xc, yc, r in zip(x_px, y_px, r_px):
        # Circular region for intensity measurements
        rr, cc = disk((yc, xc), r, shape=(H, W))
        if len(rr) > 0:
            values = arr[rr, cc]
            mean_list.append(values.mean())
            sd_list.append(values.std())
        else:
            mean_list.append(np.nan)
            sd_list.append(np.nan)

    # Modified dataframe
    df_out = df.copy()
    df_out["x_pixel"] = x_px
    df_out["y_pixel"] = y_px
    df_out["sigma_pixel"] = r_px
    df_out["foci_MFI"] = mean_list
    df_out["foci_SD"] = sd_list
    df_out["pixel_size_nm"] = px_size_nm

    # Plot only foci centers
    if plot:
        draw_foci(
            image=image,
            df=df_out,
            showplot=False,
            save_image=True,
            save_path=save_path
        )

    return df_out

#---------------------
# MAIN FUNCTION
#---------------------
def main():
    # Ask user about the path to the directory with images and foci.csv files
    dir_images = check_directory(input("Enter pathway to the directory with the images: "))
    dir_foci = check_directory(input("Enter pathway to the directory with ThunderSTORM output in .csv format): "))

    # Pixel size override, for ThunderSTORM CSVs produced without a protocol file
    px_answer = input(
        "ThunderSTORM pixel size in nm [Enter = read from protocol file, else from TIFF]: "
    ).strip()
    px_override = float(px_answer) if px_answer else None

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
    and f.lower().endswith(".tif") and "_roi_" in f.lower()
    ]

    print(f"Number of founded images is {len(paths_images)}")

    # List of paths to the ThunderSTORM CSVs (<image>_foci.csv, written by run_thunderstorm.py)
    suffix = "_foci.csv"
    paths_foci_csv = [
        os.path.join(dir_foci, f)
        for f in os.listdir(dir_foci)
        if os.path.isfile(os.path.join(dir_foci, f))
        and f.lower().endswith(suffix)
        ]

    print(f"Number of founded .csv files is {len(paths_foci_csv)}")

    # Create dictionaries
    img_by_key = {filename(image_name).replace(" ", "_"): image_name for image_name in paths_images} # dictionary {image name w/o ext: image path}

    csv_by_key = {os.path.basename(csv_name)[:-len(suffix)].replace(" ", "_"): csv_name for csv_name in paths_foci_csv} # dictionary {image name: csv path}

    # Pair only images that have a CSV (and vice versa); report the rest
    keys = sorted(img_by_key.keys() & csv_by_key.keys())
    for k in sorted(img_by_key.keys() - csv_by_key.keys()):
        print(f"No ThunderSTORM CSV for image: {k}")
    for k in sorted(csv_by_key.keys() - img_by_key.keys()):
        print(f"No image for ThunderSTORM CSV: {k}")
    combined = {k: (img_by_key[k], csv_by_key[k]) for k in keys} # dictionary {file_name: (path_to_image, path_to_foci_csv)}

    n_images = len(combined)
    print(f"Founded {n_images} pairs of image.tif : foci.csv files.")

    # Iteration through combined dictioanry
    for name, (img_path, csv_path) in combined.items():
        try:
            # Open image
            image = Image.open(img_path)

            # Read foci.csv file
            df = pd.read_csv(csv_path)

            # Pixel size ThunderSTORM used: override > protocol file > TIFF calibration
            px_tiff = tiff_pixel_size_nm(image)
            px_protocol = protocol_pixel_size_nm(csv_path)
            if px_override is not None:
                px, source = px_override, "user input"
            elif px_protocol is not None:
                px, source = px_protocol, "ThunderSTORM protocol"
            elif px_tiff is not None:
                px, source = px_tiff, "TIFF calibration (no protocol file found)"
            else:
                raise ValueError("pixel size unknown: no protocol file, no TIFF calibration; enter it at the prompt")
            if px_tiff is not None and abs(px - px_tiff) / px_tiff > PIXEL_SIZE_TOLERANCE:
                print(f"  NOTE {name}: ThunderSTORM pixel size {px:.2f} nm differs from TIFF calibration {px_tiff:.2f} nm")
            print(f"  {name}: pixel size {px:.3f} nm ({source})")

            result = foci_one_image(image = image,
                                    df = df,
                                    px_size_nm = px,
                                    plot = True,
                                    save_path = os.path.join(output_dir, f"{name}_foci.png"),
                                    name = name)

            # Save the result to a new .csv file
            result_csv_path = os.path.join(output_dir, f"{name}_foci_mapped.csv")
            result.to_csv(result_csv_path, index=False)

            print(f"Sucessfully processed image {name}.")

        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

if __name__ == "__main__":
    main()
