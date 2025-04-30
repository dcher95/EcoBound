# Testing script to compare outputs for species to eBird maps

# Assumption: If species shows presence in a location -> present. (If 1 and 0 --> 1)

import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
import json
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from dataset import MapDataset
from model import SDM

from config import config


def generate_binary_presence_dataset(csv_path, output_csv_path=None):
    """
    Generates a binary presence dataset for species at each unique location.

    Parameters:
    - csv_path (str): Path to the input eBird CSV file.
    - output_csv_path (str, optional): If provided, the binary dataset will be saved to this file.

    Returns:
    - pd.DataFrame: DataFrame with binary presence (1 for present, 0 for absent) for each species,
      grouped by unique latitude and longitude.
    """
    # Load the CSV file. Adjust index_col if your CSV includes an extra index column.
    df = pd.read_csv(csv_path, index_col=0)
    
    # Define metadata columns that are not species data.
    metadata_cols = [
        'checklist_id', 'observer_id', 'observation_date', 'hours_of_day',
        'latitude', 'longitude', 'observation_count', 'effort_hours',
        'effort_distance_km', 'effort_speed_kmph', 'number_observers'
    ]
    
    # Exclude the metadata and species_richness columns to get species columns.
    species_cols = [col for col in df.columns if col not in metadata_cols + ['species_richness']]
    
    # Group by location (latitude and longitude) and aggregate species counts using max.
    # Then convert any positive count to 1 (presence) and 0 remains 0 (absence).
    binary_presence = (
        df.groupby(['latitude', 'longitude'])[species_cols]
          .max()
          .applymap(lambda x: 1 if x > 0 else 0)
          .reset_index()
    )

    # Necessary for MapDataset class
    binary_presence.rename(columns = {'latitude' : 'lat',
                                      'longitude' : 'lon'}, inplace = True)
    
    # Optionally, save the result to a CSV file.
    if output_csv_path:
        binary_presence.to_csv(output_csv_path, index=False)
    
    return binary_presence, species_cols

def filter_species_by_presence(json_path, threshold_percent):
    """
    Reads a species summary JSON file and filters species that have at least the specified percentage presence.

    Parameters:
        json_path (str): Path to the species summary JSON file.
        threshold_percent (float): The minimum percentage presence required (e.g., 28 for 28%).

    Returns:
        dict: A dictionary of filtered species with their metrics.
    """
    with open(json_path, 'r') as f:
        species_summary = json.load(f)

    filtered_species_dict = {
        species: metrics 
        for species, metrics in species_summary.items() 
        if metrics.get("Percent Presence", 0) >= threshold_percent
    }

    filtered_species = list(filtered_species_dict.keys())
    
    return filtered_species

def inference(model, maploader, species):

    if species:
        species_data = np.load("./data/species.npy", allow_pickle=True)
        species_index = np.where(species_data == species)[0][0]

    # Run inference
    species_probs = []
    with torch.no_grad():
        for batch in tqdm(maploader):
            loc_feats = batch.cuda()
            if species:
                logits = model(loc_feats, class_of_interest=species_index)
            else:
                logits = model(loc_feats)
            probs = torch.sigmoid(logits).cpu().numpy()
            species_probs.append(probs)

    species_probs = np.concatenate(species_probs, axis=0)

    return species_probs

def get_valid_species(binary_presence_df, inat_species_path):
    inat_species_list = np.load(inat_species_path, allow_pickle=True).tolist()
    valid_species = [sp for sp in binary_presence_df.columns if sp in inat_species_list]
    return valid_species, inat_species_list

def reorder_predictions(species_probs, valid_species, inat_species_list):
    indices = [inat_species_list.index(sp) for sp in valid_species]
    return species_probs[:, indices]

def calculate_metrics(ground_truth, predictions, species_list, threshold=0.5):
    aps, aucs, f1s = {}, {}, {}
    for i, sp in enumerate(species_list):
        y_true = ground_truth[:, i]
        y_score = predictions[:, i]
        y_pred = (y_score >= threshold).astype(int)

        if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
            # Avoid undefined AUC or AP due to lack of class variation
            continue

        try:
            aps[sp] = average_precision_score(y_true, y_score)
            aucs[sp] = roc_auc_score(y_true, y_score)
            f1s[sp] = f1_score(y_true, y_pred)
        except Exception as e:
            print(f"Skipping species {sp} due to error: {e}")

    mean_metrics = {
        "mean_ap": np.mean(list(aps.values())) if aps else 0,
        "mean_auc": np.mean(list(aucs.values())) if aucs else 0,
        "mean_f1": np.mean(list(f1s.values())) if f1s else 0
    }
    return aps, aucs, f1s, mean_metrics


def evaluate_species_subset(
    binary_presence_df, species_probs, valid_species, inat_species_list, 
    ebird_species_data=None, threshold_percent=None
):
    if threshold_percent is not None:
        # Filter species by checklist percentage
        filtered_species = filter_species_by_presence(ebird_species_data, threshold_percent)
        species_subset = [sp for sp in valid_species if sp in filtered_species]
    else:
        # Use full set
        species_subset = valid_species

    if not species_subset:
        return None, None, species_subset

    indices = [valid_species.index(sp) for sp in species_subset]
    ground_truth = binary_presence_df[species_subset].values
    predictions = species_probs[:, indices]

    aps, aucs, f1s, mean_metrics = calculate_metrics(ground_truth, predictions, species_subset)
    return aps, aucs, f1s, mean_metrics

def save_metrics(metrics_dict, experiment_name, output_dir="./outputs/metrics/"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{experiment_name}_metrics_by_threshold.json")
    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metric outputs saved to {output_path}")

def main():
    experiment_name = config.experiment_name
    testing_df_path = config.testing_df_path
    species = None
    model_path = f"./models/{experiment_name}.ckpt"
    cross_species_comparison = True

    # Load dataset
    binary_presence_df, _ = generate_binary_presence_dataset(
        testing_df_path, "/data/cher/EcoBound/data/eBird/binary_presence.csv"
    )
    mapdataset = MapDataset(sampled_data=binary_presence_df)
    maploader = torch.utils.data.DataLoader(
        mapdataset, batch_size=config.batch_size, shuffle=False, num_workers=16
    )

    # load_model_and_predict
    model = SDM.load_from_checkpoint(model_path)
    model.cuda().eval()
    species_probs = inference(model, maploader, species)

    # Species setup
    valid_species, inat_species_list = get_valid_species(
        binary_presence_df, "/data/cher/EcoBound/data/species.npy"
    )
    species_probs_filtered = reorder_predictions(species_probs, valid_species, inat_species_list)

    metrics_by_threshold = {}

    # Full dataset
    aps_all, aucs_all, f1s_all, mean_all = evaluate_species_subset(
        binary_presence_df, species_probs_filtered, valid_species, inat_species_list
    )
    
    print(f"Full Dataset — mAP: {mean_all['mean_ap']:.3f}, AUC: {mean_all['mean_auc']:.3f}, F1: {mean_all['mean_f1']:.3f}")
    metrics_by_threshold["full"] = {
        "mean_metrics": mean_all,
        "aps": aps_all,
        "aucs": aucs_all,
        "f1s": f1s_all
    }

    # Cross-species evaluation
    if cross_species_comparison:
        num_species = len(valid_species)
        cross_species_metrics = {}

        for i, pred_sp in enumerate(valid_species):
            for j, true_sp in enumerate(valid_species):
                if i == j:
                    continue  # Skip same-species unless you want self-evals too

                y_score = species_probs_filtered[:, i]
                y_true = binary_presence_df[true_sp].values
                y_pred = (y_score >= 0.5).astype(int)

                if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
                    continue  # No class variation — skip

                try:
                    ap = average_precision_score(y_true, y_score)
                    auc = roc_auc_score(y_true, y_score)
                    f1 = f1_score(y_true, y_pred)
                    if ap > 0.3:
                        cross_species_metrics[f"{pred_sp}_on_{true_sp}"] = {
                            "ap": ap,
                            "auc": auc,
                            "f1": f1
                        }
                        print(f"{pred_sp} on {true_sp} — AP: {ap:.3f}, AUC: {auc:.3f}, F1: {f1:.3f}")
                except Exception as e:
                    print(f"Skipping {pred_sp} on {true_sp}: {e}")
                    continue

        # Save cross-species metrics separately
        save_metrics(cross_species_metrics, experiment_name + "_cross_species")


    # Thresholded evaluations
    else:
        ebird_species_data = '/data/cher/EcoBound/data/eBird/species_summary.json'
        thresholds = [0.5, 1.0, 2.0, 5.0]
        for threshold in thresholds:
            aps, aucs, f1s, mean_metrics = evaluate_species_subset(
                binary_presence_df, species_probs_filtered, valid_species,
                inat_species_list, ebird_species_data, threshold
            )
            if aps is not None:
                print(f"{threshold}% Threshold — mAP: {mean_metrics['mean_ap']:.3f}, "
                    f"AUC: {mean_metrics['mean_auc']:.3f}, F1: {mean_metrics['mean_f1']:.3f}")
                metrics_by_threshold[f"{threshold}_percent"] = {
                    "mean_metrics": mean_metrics,
                    "aps": aps,
                    "aucs": aucs,
                    "f1s": f1s
                }
            else:
                print(f"No species met the {threshold}% presence threshold.")


        # Save results
        save_metrics(metrics_by_threshold, experiment_name)

if __name__ == "__main__":
    main()