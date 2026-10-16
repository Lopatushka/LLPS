import pandas as pd

def table_aggregation(path):
    df = pd.read_csv(path)
    
    image_id = df["image_id"][0]
    n_foci = df.shape[0]
    
    area = df["area"].mean()
    equivalent_diameter_area = df["equivalent_diameter_area"].mean()
    eccentricity = df["eccentricity"].mean()
    solidity = df["solidity"].mean()
    perimeter = df["perimeter"].mean()
    mean_intensity = df["mean_intensity"].mean()
    max_intensity = df["max_intensity"].mean()
    min_intensity = df["min_intensity"].mean()
    
    row = {
        "image_id": image_id,
        "n_foci": n_foci,
        "area": area,
        "equivalent_diameter_area": equivalent_diameter_area,
        "eccentricity": eccentricity,
        "solidity": solidity,
        "perimeter": perimeter,
        "mean_intensity": mean_intensity,
        "max_intensity": max_intensity,
        "min_intensity": min_intensity,
        
    }
    
    return row
    
    