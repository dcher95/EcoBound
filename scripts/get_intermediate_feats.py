import pandas as pd
import torch
import numpy as np
import os
from tqdm import tqdm

from dataset import MapDataset

from model import SDM

from config import config

def main():
    # Inputs
    experiment_name = config.experiment_name

    ################

    # Model for inference
    model = SDM.load_from_checkpoint(f"./models/{experiment_name}.ckpt")
    model.cuda().eval()

    # Densely sampled dataset
    mapdataset = MapDataset()
    maploader = torch.utils.data.DataLoader(mapdataset, batch_size=config.batch_size, shuffle=False, num_workers=16)

    # Get intermediate features
    features_list = []
    with torch.no_grad():
        for batch in tqdm(maploader):
            loc_feats = batch.cuda()
            # Return Intermediate features
            features = model.loc_encoder(loc_feats, return_feats=True)
            features_list.append(features.cpu().numpy())

    # Concatenate all feature arrays if needed
    features_all = np.concatenate(features_list, axis=0)

    os.makedirs("./outputs/loc_features", exist_ok = True) 
    np.save(f'./outputs/loc_features/{experiment_name}.npy', features_all)

if __name__ == "__main__":
    main()