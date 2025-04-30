from config import (
    species_to_plot,
    start_lat, start_lon, avail_trips,
    lambda_param, analytic_df
)
from data_loader import load_training_species, load_data
from sampling import search_loop_multi

# set seed

# Load training species
training_species = load_training_species(species_to_plot)

# load training data
analytic_df = load_data()

# Sampling methods to test
sampling_methods = ["random", "uncertainty", "costaware"]

# Storage for results
all_weights = {}
all_map_scores = {}
all_auc_scores = {}
trip_stops = {}

# Loop through sampling methods
for method in sampling_methods:
    print(f"\nRunning method: {method}")
    weights, map_scores, auc_scores, stops = search_loop_multi(
        species_list=training_species,
        method=method,
        sampds=analytic_df.copy(),  # important to avoid modifying original
        lambda_val=lambda_param
    )

    # Save results
    all_weights[method] = weights
    all_map_scores[method] = map_scores
    all_auc_scores[method] = auc_scores
    trip_stops[method] = stops
