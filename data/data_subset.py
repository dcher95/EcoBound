import pandas as pd
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats.qmc import Sobol

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["species", "decimalLatitude", "decimalLongitude"])
    return df

def limit_by_species(df, max_per_species):
    return df.groupby("species").head(max_per_species).reset_index(drop=True)

def apply_bounding_box(df, bbox):
    min_lon, max_lon, min_lat, max_lat = bbox
    return df[
        (df["decimalLongitude"] >= min_lon) & (df["decimalLongitude"] <= max_lon) &
        (df["decimalLatitude"] >= min_lat) & (df["decimalLatitude"] <= max_lat)
    ].copy()

def sobol_sample_by_location(df, bbox, n_samples):
    min_lon, max_lon, min_lat, max_lat = bbox
    sobol_engine = Sobol(d=2, scramble=True)
    samples = sobol_engine.random(n=n_samples)
    
    # Scale Sobol points to lat/lon
    longitudes = samples[:, 0] * (max_lon - min_lon) + min_lon
    latitudes = samples[:, 1] * (max_lat - min_lat) + min_lat
    sobol_points = np.stack([latitudes, longitudes], axis=1)

    # For each Sobol point, find the closest observation
    coords = df[["decimalLatitude", "decimalLongitude"]].to_numpy()
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(coords)
    distances, indices = nn.kneighbors(sobol_points)
    selected_indices = np.unique(indices.flatten())
    
    return df.iloc[selected_indices].copy()

def main():

    # === CONFIGURATION ===
    INPUT_CSV = "./data/gbif_full_filtered.csv"
    MAX_PER_SPECIES = 1000
    BOUNDING_BOX = (-90.68099, -90.09099, 38.45601, 38.88601)  # (min_lon, max_lon, min_lat, max_lat)
    SOBOL_SAMPLES = 524288 # 2 ** 19
    OUTPUT_DIR = "./data/subsets"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(INPUT_CSV)

    # just in case
    np.random.seed(42)

    # Strategy 1: Species-limited
    df_species_limited = limit_by_species(df, MAX_PER_SPECIES)
    df_species_limited.to_csv(os.path.join(OUTPUT_DIR, f"iNat_species_limited_max{MAX_PER_SPECIES}.csv"), index=False)

    # Strategy 2: Sobol sample over bounding box. 1000 per species.
    df_bbox = apply_bounding_box(df, BOUNDING_BOX)
    df_sobol_sample = sobol_sample_by_location(df_bbox, BOUNDING_BOX, SOBOL_SAMPLES)
    df_sobol_sample.to_csv(
        os.path.join(OUTPUT_DIR, f"iNat_sobol_sample_n{SOBOL_SAMPLES}.csv"), index=False
    )

    # I don't think this makes sense. So excluding.
    # # Strategy 3: Combined species limit + Sobol
    # df_species_limited_bbox = limit_by_species(df_bbox, MAX_PER_SPECIES)
    # df_combined = sobol_sample_by_location(df_species_limited_bbox, BOUNDING_BOX, SOBOL_SAMPLES)
    # df_combined.to_csv(
    #     os.path.join(OUTPUT_DIR, f"iNat_combined_species_sobol_n{SOBOL_SAMPLES}.csv"), index=False
    # )

if __name__ == "__main__":
    main()
