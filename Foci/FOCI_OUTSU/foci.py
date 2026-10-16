import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from skimage import io, filters, morphology, segmentation, segmentation, exposure
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage import io, filters, morphology, measure, segmentation, segmentation, exposure

from pathlib import Path
import traceback

def process_image(image_path, threshold_factor=3,  min_area=2, min_distance=5, plot=True, save_path=None):
    # Download image
    image = io.imread(image_path)
    image_name = Path(image_path).stem
    
    threshold = filters.threshold_otsu(image)
    
    binary_mask = image > threshold * threshold_factor

    clean_mask = morphology.remove_small_objects(
        binary_mask,
        min_size=min_area
    )

    image_float = image.astype(np.float32)
    distance = ndi.distance_transform_edt(clean_mask)

    # Detect candidate centers
    coordinates = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=clean_mask
    )

    markers = np.zeros_like(clean_mask, dtype=np.int32)

    for i, (y, x) in enumerate(coordinates, start=1):
        markers[y, x] = i

    labels = segmentation.watershed(
        -distance,
        markers,
        mask=clean_mask
    )
    
    properties = measure.regionprops_table(
    labels,
    intensity_image=image_float,
    properties=[
        "label",
        "area",
        "centroid",
        "equivalent_diameter_area",
        "eccentricity",
        "solidity",
        "perimeter",
        "mean_intensity",
        "max_intensity",
        "min_intensity"
    ]
    )

    df = pd.DataFrame(properties)

    # Circularity: 1 means approximately circular
    df["circularity"] = (
        4 * np.pi * df["area"]
        / (df["perimeter"] ** 2 + 1e-8)
    )

    df["image_id"] = image_name


    print("Total detected objects:", len(df))

    # Plot or save figure
    if plot or save_path is not None:

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].imshow(clean_mask, cmap="gray")
        axes[0].set_title("Binary mask")

        axes[1].imshow(distance, cmap="magma")
        axes[1].set_title("Distance transform")

        axes[2].imshow(
            segmentation.mark_boundaries(
                image_float / (image_float.max() + 1e-8),
                labels
            )
        )
        axes[2].set_title(f"Watershed: {labels.max()} objects")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        # Save figure
        if save_path is not None:
            save_path = Path(save_path)

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )
            
    return df

def main():
    input_dir = Path(input("Enter input folder path: ").strip().strip('"'))
    output_dir = Path(input("Enter output folder path: ").strip().strip('"'))

    # Validate input folder
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    # Create output folder if necessary
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all TIFF images
    paths = sorted(
        list(input_dir.glob("*.tif")) +
        list(input_dir.glob("*.tiff"))
    )

    print(f"Found {len(paths)} images")
    
    for path in paths:
        try:            
            # Output figure path
            save_path = output_dir / f"{path.stem}_segmentation.png"
            csv_path = output_dir / f"{path.stem}_segmentation.csv"
            
            result = process_image(
                image_path=path,
                threshold_factor=3, 
                min_area=10,
                min_distance=5,
                plot=True,
                save_path=save_path,
                )
            
            result.to_csv(csv_path, index=False)
            
            print(
                f"Processed: {path.name}"
            )

            
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            traceback.print_exc()
            continue
        
    print("Processing completed!")
    
if __name__ == "__main__":
    main()
            