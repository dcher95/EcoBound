import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from get_intermediate_feats import extract_features
from inference import inference
from testing import get_valid_species, reorder_predictions

from dataset import MapDataset
from model import SDM

from config import config

def main():
    experiment_name = config.experiment_name
    testing_df_path = config.testing_df_path
    species = None
    all_species = True
    
    model_path = f"./models/{experiment_name}.ckpt"

    # generated in testing
    binary_presence_path = "/data/cher/EcoBound/data/eBird/binary_presence.csv"

    # Load ground truth dataset (assumes a CSV with lat, long, and species presence columns)
    binary_presence_df = pd.read_csv(binary_presence_path)
    
    # Create a dataset for the map loader.
    mapdataset = MapDataset(sampled_data=binary_presence_df)
    maploader = torch.utils.data.DataLoader(
        mapdataset, batch_size=32, shuffle=False, num_workers=4)  # Adjust batch_size and num_workers as needed

    # Load the pretrained model (assumes an SDM model with loc_encoder and forward method)
    model = SDM.load_from_checkpoint(model_path)
    model.cuda().eval()

    # --- Feature Extraction ---
    features_all = extract_features(model, maploader)

    # --- Inference: Generate Species Probabilities ---
    species_probs = inference(model, maploader, species)

    # --- Filter and Reorder Species ---
    valid_species, inat_species_list = get_valid_species(binary_presence_df, "./data/species.npy")
    species_probs_filtered = reorder_predictions(species_probs, valid_species, inat_species_list)

    # --- Build the Combined DataFrame ---
    # Start with lat and long columns from the ground truth
    if not {'lat', 'lon'}.issubset(binary_presence_df.columns):
        raise ValueError("The input DataFrame must contain 'lat' and 'lon' columns.")
    df_output = binary_presence_df[['lat', 'lon']].copy()

    # Include species presence for valid species
    df_presence = binary_presence_df[valid_species]

    # Build a DataFrame for species probabilities using pd.concat
    if all_species:
        prob_col_names = [f"prob_{sp}" for sp in inat_species_list]
        df_probs = pd.DataFrame(species_probs, columns=prob_col_names, index=df_output.index)
    else:
        prob_col_names = [f"prob_{sp}" for sp in valid_species]
        df_probs = pd.DataFrame(species_probs_filtered, columns=prob_col_names, index=df_output.index)

    # Build a DataFrame for the features column (each row gets a list of features)
    df_features = pd.DataFrame({"features": list(features_all)})

    # Concatenate all columns at once along axis 1
    newframe = pd.concat([df_output, df_presence, df_probs, df_features], axis=1)

    # Optionally, get a defragmented copy of the new DataFrame
    newframe = newframe.copy()

    # Save the combined DataFrame
    os.makedirs("./outputs/active_loop", exist_ok=True)
    output_csv_path = f"./outputs/active_loop/{experiment_name}_combined_all_species.csv"
    newframe.to_csv(output_csv_path, index=False)
    print(f"Combined dataframe saved to {output_csv_path}")

if __name__ == "__main__":
    main()
