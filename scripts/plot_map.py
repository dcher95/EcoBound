import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

def load_data(experiment_name, species_file, coords_file, probs_file):
    species_names = np.load(species_file, allow_pickle=True)
    coords_df = pd.read_csv(coords_file)
    species_probs = np.load(probs_file, allow_pickle=True)
    
    prob_df = pd.DataFrame(species_probs, columns=[f"prob_{name}" for name in species_names])
    results_df = pd.concat([coords_df, prob_df], axis=1)
    
    return results_df, species_names

def generate_species_heatmap(results_df, species_list, experiment_name, output_dir="./outputs/species_priors/maps"):
    os.makedirs(output_dir, exist_ok=True)
    
    for species in tqdm(species_list, desc = 'Processing species'):
        df = results_df[['lon', 'lat', f"prob_{species}"]].copy()
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
        gdf = gdf.to_crs("EPSG:3857")  # Convert to Web Mercator
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        hb = ax.hexbin(gdf.geometry.x, gdf.geometry.y, C=df[f"prob_{species}"], 
                       gridsize=100, cmap="YlGnBu", reduce_C_function=np.mean, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(hb, ax=ax, label=f"Probability of {species}")
        
        # Add OpenStreetMap basemap
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=12, alpha=0.5)
        
        # Labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{species} Probability Heatmap")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save plot
        output_path = os.path.join(output_dir, f"{species}-{experiment_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Saved: {output_path}")

def main():
    experiment_name = 'STL-loc-an_full-wgt'
    species_list = ["Sciurus carolinensis", "Danaus plexippus", "Cardinalis cardinalis", "Quercus alba"]
    results_df, species_names = load_data(
        experiment_name, 
        "./data/species.npy", 
        "./data/densely_sampled_pts.csv", 
        f"./outputs/species_priors/{experiment_name}.npy"
    )
    generate_species_heatmap(results_df, species_list, experiment_name)

if __name__ == "__main__":
    main()