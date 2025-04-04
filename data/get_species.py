import pandas as pd
import numpy as np

def generate_species_list():
    observation_path = "data/gbif_full_filtered.csv"

    # Load data
    observation_df = pd.read_csv(observation_path)

    # Get unique species and counts
    species_counts = observation_df['species'].value_counts().to_dict()
    print(f"Found {len(species_counts)} unique species")

    # Save species list
    np.save("data/species.npy", np.array(list(species_counts.keys())))
    print("Saved species list to species.npy")

    # Save species counts as a dictionary
    np.save("data/species_counts.npy", species_counts)
    print("Saved species counts to species_counts.npy")

if __name__ == "__main__":
    generate_species_list()
