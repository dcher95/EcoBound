import json
import argparse

from config import config

def filter_species_by_metric(file_path, metric='ap', cutoff=0.1):
    """
    Filters species from the JSON file based on the provided metric and cutoff value.

    Parameters:
        file_path (str): Path to the JSON file.
        metric (str): Metric to filter by ('ap' for average precision or 'auc' for area under curve).
        cutoff (float): The threshold value for the metric.

    Returns:
        dict: Dictionary of species and their metric values above the threshold.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Choose the appropriate key based on metric input
    if metric.lower() == 'ap':
        metric_dict = data['full'].get('aps', {})
    elif metric.lower() == 'auc':
        metric_dict = data['full'].get('aucs', {})
    else:
        raise ValueError("Metric must be either 'ap' or 'auc'")

    # Filter species with metric value above the cutoff
    filtered_species = {species: value for species, value in metric_dict.items() if value >= cutoff}
    return filtered_species

def main():
    metric = 'auc'  # or 'auc'
    cutoff = 0.6 # 0.3 for ap, 0.5 for auc

    file_path = f"/data/cher/EcoBound/outputs/metrics/{config.experiment_name}_metrics_by_threshold.json"
    out_path = f"/data/cher/EcoBound/outputs/filtered_metrics/{config.experiment_name}_filtered_species_{metric}{cutoff}.json"

    result = filter_species_by_metric(file_path, metric, cutoff)
    print("Filtered species:")
    for species, value in result.items():
        print(f"{species}: {value}")

    # Save final result to an output JSON file
    with open(out_path, 'w') as out_file:
        json.dump(result, out_file, indent=4)
    print(f"\nFiltered results saved to: {out_path}")

if __name__ == "__main__":
    main()