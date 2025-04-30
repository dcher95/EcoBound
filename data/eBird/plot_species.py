import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point

def plot_selected_species_locations(df, species_list, output_path="./"):
    """
    Plots presence/absence of species and saves them as PNG files.

    Parameters:
    - df (pd.DataFrame): DataFrame with latitude, longitude, and species columns.
    - species_list (list of str): List of species to plot.
    - output_path (str): Directory to save PNG files to.
    """

    # Create geometry column
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    for species in species_list:
        if species not in gdf.columns:
            print(f"Species '{species}' not found in DataFrame columns. Skipping.")
            continue

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot absence
        gdf[gdf[species] == 0].plot(
            ax=ax, markersize=5, color='grey', label='Absence', alpha=0.6
        )

        # Plot occurrence
        gdf[gdf[species] == 1].plot(
            ax=ax, markersize=50, color='blue', label='Occurrence', alpha=0.3
        )

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        ax.legend()
        ax.set_title(species)

        # Save to file
        filename = os.path.join(output_path, f"{species.replace(' ', '_')}.png")
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Saved: {filename}")


if __name__ == "__main__":
    ebird_df = pd.read_csv("/data/cher/EcoBound/data/eBird/binary_presence.csv")
    species_to_plot = [
    "Aix galericulata", "Cardinalis cardinalis", "Turdus migratorius", "Cyanocitta cristata", "Haemorhous mexicanus",
    "Zenaida macroura", "Piranga ludoviciana", "Sturnus vulgaris", "Passer domesticus", "Dryobates pubescens", "Baeolophus bicolor", "Melanerpes carolinus",
    "Passer montanus", "Spinus tristis", "Branta canadensis", "Poecile carolinensis", "Sitta carolinensis", "Thryothorus ludovicianus",
    "Sialia sialis", "Junco hyemalis", "Agelaius phoeniceus", "Anas platyrhynchos", "Zonotrichia albicollis", "Passerina ciris", "Quiscalus quiscula",
    "Quiscalus mexicanus", "Ardea herodias", "Corvus brachyrhynchos"]
    
    plot_selected_species_locations(ebird_df, species_to_plot, output_path="/data/cher/EcoBound/outputs/eBird")
