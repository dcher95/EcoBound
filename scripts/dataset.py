import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict


def sinusoidal_encoding(lon: float, lat: float, bounds: Tuple[float, float, float, float]) -> torch.FloatTensor:
    """
    Computes sinusoidal encoding for given longitude and latitude values.
    
    Args:
        lon (float): Longitude value.
        lat (float): Latitude value.
        bounds (Tuple[float, float, float, float]): Geographic bounds (lon_min, lon_max, lat_min, lat_max).
    
    Returns:
        torch.FloatTensor: Encoded location features (4D tensor).
    """
    feats_lon = 2 * ((lon - bounds[0]) / (bounds[1] - bounds[0])) - 1
    feats_lat = 2 * ((lat - bounds[2]) / (bounds[3] - bounds[2])) - 1
    
    return torch.FloatTensor([
        np.sin(np.pi * feats_lon / 2),
        np.cos(np.pi * feats_lon / 2 + np.pi / 2),
        np.sin(np.pi * feats_lat / 2),
        np.cos(np.pi * feats_lat / 2 + np.pi / 2)
    ])

class LocationDataset(Dataset):
    def __init__(self, 
                obs_file: str,
                species_file: str = "./data/species.npy",
                coords_file: str = "./data/st_louis_coords.csv",
                bounds: Optional[Tuple[float, float, float, float]] = (-90.6809899999999942, -90.0909899999996924, 38.4560099999999991, 38.8860099999999136), 
                transform=None):
        """
        Dataset for species observations, encoding locations using sinusoidal features.

        Args:
            csv_file (str): Path to the CSV file containing species observations.
            bounds (Optional[Tuple[float, float, float, float]]): Geographic bounds (lon_min, lon_max, lat_min, lat_max).
                Defaults to St. Louis coordinates if not provided.
        """
        self.obs = pd.read_csv(obs_file)
        self.bounds = bounds
        self.stl_coords = pd.read_csv(coords_file)

        species_data = np.load(species_file, allow_pickle=True)
        self.species_to_index = {species_data[i] : i  for i in range(len(species_data))}

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Retrieves an observation and its encoded location features.

        Args:
            idx (int): Index of the observation.

        Returns:
            Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
                - Species class index
                - Sinusoidal encoding of observed location
                - Sinusoidal encoding of a randomly sampled location
        """
        lon, lat = float(self.obs.iloc[idx]['decimalLongitude']), float(self.obs.iloc[idx]['decimalLatitude'])
        feats = sinusoidal_encoding(lon, lat, self.bounds)

        # Generate random coordinates for pseudo-negative sampling
        rand_lon = np.random.uniform(self.bounds[0] + 0.01, self.bounds[1] - 0.01)
        rand_lat = np.random.uniform(self.bounds[2] + 0.01, self.bounds[3] - 0.01)
        rand_feats = sinusoidal_encoding(rand_lon, rand_lat, self.bounds)

        species = self.obs.iloc[idx]['species']
        species_class = self.species_to_index.get(species)

        return torch.LongTensor([species_class]), feats, rand_feats

# Used for inference!    
class MapDataset(Dataset):
    def __init__(self, 
                 sampled_csv : str = '/data/cher/EcoBound/data/densely_sampled_pts.csv', 
                 bounds: Optional[Tuple[float, float, float, float]] = (-90.6809899999999942, -90.0909899999996924, 38.4560099999999991, 38.8860099999999136), 
                ):
        self.coords = pd.read_csv(sampled_csv)
        # self.coords = self.coords[self.coords["lon"] != max(self.coords["lon"])]
        # self.coords = self.coords[self.coords["lat"] != max(self.coords["lat"])]
        # self.coords = self.coords[self.coords["lon"] != min(self.coords["lon"])]
        # self.coords = self.coords[self.coords["lat"] != min(self.coords["lat"])]
        self.bounds = bounds
    
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):

        lon = float(self.coords.iloc[idx]["lon"])
        lat = float(self.coords.iloc[idx]["lat"])
        
        feats = sinusoidal_encoding(lon, lat, self.bounds)

        return feats
    
if __name__ == "__main__":
    dataset = LocationDataset("gbif_full_filtered.csv")
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
