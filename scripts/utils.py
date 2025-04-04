import numpy as np
import torch

def get_species_weights(num_obs, species_counts, species_weights_method):
    """
    Compute species weights given the total number of observations and a tensor of species counts.
    
    Parameters:
        num_obs (float): Total number of observations.
        species_counts (Tensor): Tensor containing counts for each species.
        species_weights_method (str): One of:
            "inversely_proportional", "inversely_proportional_clipped",
            "inversely_proportional_sqrt", "uniform",
            "inversely_proportional_not_normalized".
    
    Returns:
        Tensor: Computed species weights.
    """
    if species_weights_method == "inversely_proportional":
        species_weights = num_obs / (species_counts + 1e-5)
    elif species_weights_method == "inversely_proportional_clipped":
        species_weights = num_obs / (species_counts + 1e-5)
        species_weights = np.clip(species_weights, 0.05, 20)
    elif species_weights_method == "inversely_proportional_sqrt":
        species_weights = np.sqrt(num_obs / (species_counts + 1e-5))
    elif species_weights_method == "uniform":
        species_weights = torch.ones_like(species_counts, device=species_counts.device)
    elif species_weights_method == "inversely_proportional_not_normalized":
        species_weights = 1 / (species_counts + 1e-5)
    else:
        raise ValueError(
            "species_weights_method must be 'inversely_proportional', "
            "'inversely_proportional_clipped', 'inversely_proportional_sqrt', "
            "'uniform' or 'inversely_proportional_not_normalized'"
        )
    return species_weights

def load_species_weights(species_file, species_counts_file, species_weights_method):
    """
    Load species names and counts from .npy files, create a species-to-index mapping, 
    build a tensor of species counts, and compute the species weights.
    
    Parameters:
        species_file (str): Path to the .npy file containing species names.
        species_counts_file (str): Path to the .npy file containing a dictionary of species counts.
        species_weights_method (str): Method to compute species weights.
        
    Returns:
        species_to_index (dict): Mapping from species name to index.
        species_weights (Tensor): Precomputed species weights.
        num_classes (int): Number of species.
    """
    # Load species names and create mapping.
    species_data = np.load(species_file, allow_pickle=True)
    species_to_index = {species_data[i]: i for i in range(len(species_data))}
    num_classes = len(species_data)
    
    # Load species counts (assumed to be a dictionary).
    species_counts_data = np.load(species_counts_file, allow_pickle=True).item()
    
    # Create a tensor for species counts using the name-to-index mapping.
    species_counts_tensor = torch.zeros(num_classes, dtype=torch.float32)
    for species_name, count in species_counts_data.items():
        idx = species_to_index[species_name]
        species_counts_tensor[idx] = count

    # Compute total number of observations.
    num_obs = sum(species_counts_data.values())
    
    # Compute species weights.
    species_weights = get_species_weights(num_obs, species_counts_tensor, species_weights_method)
    
    return species_to_index, species_weights, num_classes
