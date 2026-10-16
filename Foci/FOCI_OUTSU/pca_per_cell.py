import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def one_pca(df_path, safe_path):
    # Read data as df
    df = pd.read_csv(df_path,  sep=",", encoding="cp1252")
    
    # Features for classification
    features = [
        "area",
        "equivalent_diameter_area",
        "eccentricity",
        "perimeter",
        "solidity",
        "mean_intensity",
        "max_intensity",
        "circularity",
    ]
    
    # Convert features to numeric
    for feature in features:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )
    
    # Select objects with known classes and complete features
    pca_df = df.dropna(subset=features).copy()
    
    if len(pca_df) < 2:
        raise ValueError("Not enough valid objects for PCA")

    # Select features and scale them 
    X = pca_df[features]
    X_scaled = StandardScaler().fit_transform(X)

    # RUN PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df["PC1"] = X_pca[:, 0]
    pca_df["PC2"] = X_pca[:, 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot grapg
    plt.figure(figsize=(9, 7))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        color="lightblue",
        s=55,
        alpha=0.7,
        ax=ax,
    )

    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
    )

    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    )

    ax.set_title(f"PCA — {Path(df_path).stem}")
    
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    
    ax.grid(alpha=0.15)

    plt.tight_layout()
    
    fig.savefig(
        safe_path,
        dpi=300,
        bbox_inches="tight"
    )
         
    plt.close(fig)
    
    return pca_df, pca
     
    
def main():
    input_dir = Path(input("Enter input folder path: ").strip().strip('"'))
    output_dir = Path(input("Enter output folder path: ").strip().strip('"'))
    
    # Validate input folder
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    # Create output folder if necessary
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV images
    paths = sorted(
        list(input_dir.glob("*.csv"))
    )
    
    print(f"Found {len(paths)} files")
    
    for path in paths:
            try:            
                # Output figure path
                save_path = output_dir / f"{path.stem}_pca.png"
                
                pca_df, pca = one_pca(df_path=path, safe_path=save_path)
                
                print(
                f"Processed: {path.name}"
                )
            except Exception as e:
                print(f"Error processing {path.name}: {e}")
                continue
    
    print("PCA processing completed!")
    
if __name__ == "__main__":
    main()