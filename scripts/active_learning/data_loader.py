import json
import pandas as pd
import numpy as np

# TODO: Should also add the analytic_df creation here

def load_training_species(exclude_species):
    with open('/Users/andrew.d.tippinmg/Library/CloudStorage/Box-Box/EcoBound/metrics/STL-train_no_overlap-loc-an_full-128-1024_filtered_species_ap0.3.json', 'r') as f:
        data = json.load(f)
    return [species for species, val in data.items() if val > 0.3 and species not in exclude_species]

# Load data
def load_data():
    coords_df = pd.read_csv('iNaturalist/densely_sampled_pts.csv')
    species_names = np.load("iNaturalist/species.npy", allow_pickle=True)
    species_probs = np.load("species_priors/STL-train_no_overlap-loc-an_full-128-1024.npy", allow_pickle=True)
    dansds = pd.read_csv('AT_intermediate_ds/STL-train_no_overlap-loc-an_full-128-1024_combined.csv')

    # Combine for mapping
    prob_df = pd.DataFrame(species_probs, columns=[f"prob_{name}" for name in species_names])
    mapping_df = pd.concat([coords_df, prob_df], axis=1)
    prob_cols_dansds = [col for col in dansds.columns if col.startswith("prob_")]
    prob_cols_mapping_df = [col for col in mapping_df.columns if col.startswith("prob_")]
    cols_to_drop = [col for col in prob_cols_mapping_df if col not in prob_cols_dansds]
    mapping_df.drop(columns=cols_to_drop, inplace=True)

    # Final analytic DataFrame
    analytic_df = dansds.copy()

    return analytic_df