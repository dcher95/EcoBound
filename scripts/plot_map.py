import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

from dataset import MapDataset
import torch
from config import config

def load_data(species_file, probs_file):

    # load coords from MapDataset to ensure consistency with inference.

    mapdataset = MapDataset()
    coords_df = mapdataset.coords.reset_index(drop=True)[['lon', 'lat']]

    species_names = np.load(species_file, allow_pickle=True)
    species_probs = np.load(probs_file, allow_pickle=True)
    
    prob_df = pd.DataFrame(species_probs, columns=[f"prob_{name}" for name in species_names])
    results_df = pd.concat([coords_df, prob_df], axis=1)
    
    return results_df, species_names

def generate_species_heatmap(results_df, species_list, experiment_name, output_dir="./outputs/species_priors/maps"):
    output_dir += f"/{experiment_name}"
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
        output_path = os.path.join(output_dir, f"{species}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Saved: {output_path}")

def main():
    experiment_name = config.experiment_name
    # species_list = ["Sciurus carolinensis", "Danaus plexippus", "Cardinalis cardinalis", "Quercus alba"]
    # mAP > 0.4
    species_list = ['Aix galericulata', 'Cardinalis cardinalis', 'Turdus migratorius',
       'Cyanocitta cristata', 'Haemorhous mexicanus', 'Zenaida macroura',
       'Piranga ludoviciana', 'Sturnus vulgaris', 'Passer domesticus',
       'Dryobates pubescens', 'Baeolophus bicolor',
       'Melanerpes carolinus', 'Passer montanus', 'Spinus tristis',
       'Branta canadensis', 'Poecile carolinensis', 'Sitta carolinensis',
       'Thryothorus ludovicianus', 'Sialia sialis', 'Junco hyemalis']
    
    results_df, species_names = load_data(
        "./data/species.npy", 
        f"./outputs/species_priors/{experiment_name}.npy"
        
    )
    generate_species_heatmap(results_df, species_list, experiment_name)

if __name__ == "__main__":
    main()