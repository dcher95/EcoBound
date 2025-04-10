import pandas as pd
import torch
import numpy as np
import os
from tqdm import tqdm

from dataset import MapDataset

from model import SDM

from config import config

def inference(model, maploader, species):
    """Generates species probabilities from the model."""
    if species:
        species_data = np.load("./data/species.npy", allow_pickle=True)
        species_index = np.where(species_data == species)[0][0]
    species_probs = []
    with torch.no_grad():
        for batch in tqdm(maploader, desc="Running Inference"):
            loc_feats = batch.cuda()
            if species:
                logits = model(loc_feats, class_of_interest=species_index)
            else:
                logits = model(loc_feats)
            probs = torch.sigmoid(logits).cpu().numpy()
            species_probs.append(probs)
    species_probs = np.concatenate(species_probs, axis=0)
    return species_probs

def main():
    # Inputs
    experiment_name = config.experiment_name
    species = None # e.g. 'Sciurus carolinensis'

    ################

    # Model for inference
    model = SDM.load_from_checkpoint(f"./models/{experiment_name}.ckpt")
    model.cuda().eval()

    # Densely sampled dataset
    mapdataset = MapDataset()
    maploader = torch.utils.data.DataLoader(mapdataset, batch_size=config.batch_size, shuffle=False, num_workers=16)

    species_probs = inference(model, maploader, species)

    os.makedirs("./outputs/species_priors", exist_ok = True) 

    if species:
        np.save(f'./outputs/species_priors/{species}-{experiment_name}.npy', species_probs)
    else:
        np.save(f'./outputs/species_priors/{experiment_name}.npy', species_probs)

if __name__ == "__main__":
    main()