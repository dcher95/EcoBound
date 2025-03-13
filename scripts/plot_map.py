import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

#########
species = "Sciurus carolinensis"
experiment_name = "STL-loc-base"

############
# Create results df
species_names = np.load("./data/species.npy", allow_pickle=True)
coords_df = pd.read_csv('./data/densely_sampled_pts.csv')
species_probs = np.load(f"./outputs/species_priors/{experiment_name}.npy", allow_pickle=True)

prob_df = pd.DataFrame(species_probs, columns=[f"prob_{name}" for name in species_names])

results_df = pd.concat([coords_df, prob_df], axis=1)

# subset to specific species
df = results_df[['lon', 'lat', f"prob_{species}"]].copy()

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
gdf = gdf.to_crs("EPSG:3857")  # Convert to Web Mercator

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create hexbin plot with transparency (alpha)
hb = ax.hexbin(gdf.geometry.x, gdf.geometry.y, C=df[f"prob_{species}"], 
               gridsize=100, cmap="YlGnBu", reduce_C_function=np.mean, alpha=0.3)  # Adjust alpha for transparency

# Add colorbar
cb = plt.colorbar(hb, ax=ax, label=f"Probability of {species}")

# Add OpenStreetMap basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=12, alpha=0.5)  # Adjust alpha for transparency

# Labels
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"{species} Probability Heatmap")

# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])

plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")
plt.show()