import pandas as pd
import numpy as np

def generate_species_list():

    observation_path = "data/gbif_full_filtered.csv"

    # Get unique species
    observation_df = pd.read_csv(observation_path)
    unique_species = observation_df['species'].unique()
    print(f"Found {len(unique_species)} unique species")
    
    # Save to numpy file
    np.save("data/species.npy", unique_species)
    print("Saved species list to species.npy")

if __name__ == "__main__":
    generate_species_list()