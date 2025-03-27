import os
import pandas as pd
import re
import json
from config import config

def combine_ebird_data(output_path):
    # Identify files matching the pattern eBird_YYYY_MM.csv
    file_pattern = r"eBird_\d{4}_\d{2}\.csv"
    all_files = [f for f in os.listdir(output_path) if re.match(file_pattern, f)]

    dataframes = []
    species_set = set()

    # Read files and collect species names
    for file in all_files:
        df = pd.read_csv(os.path.join(output_path, file))
        species_columns = df.columns[11:]  # Species columns start after the first 11 fixed columns
        species_set.update(species_columns)
        dataframes.append(df)

    # Create a full dataset with all species
    full_species_list = sorted(species_set)  # Sort for consistency
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Ensure all species columns are present
    for species in full_species_list:
        if species not in combined_df.columns:
            combined_df[species] = 0  

    # Explicitly fill any remaining NAs in species columns with 0
    combined_df[full_species_list] = combined_df[full_species_list].fillna(0).astype(int)

    # Drop species columns that contain only zeros # Untested
    combined_df = combined_df.loc[:, (combined_df != 0).any(axis=0)]

    # Reorder columns to maintain consistency
    ordered_columns = list(combined_df.columns[:11]) + full_species_list
    combined_df = combined_df[ordered_columns]

    return combined_df
    
def create_summary_statistics(df):
    """
    Generates summary statistics for each species.
    
    Summary includes:
    - Occurrences: Number of checklists (rows) where the species was recorded (count > 0).
    - Percent Presence: Percentage of checklists where the species was observed.
    """
    species_columns = df.columns[11:]
    total_records = df.shape[0]
    summary = pd.DataFrame(index=species_columns)

    # Calculate occurrences (nonzero counts)
    summary['Occurrences'] = df[species_columns].gt(0).sum()
    # Calculate percent presence
    summary['Percent Presence'] = (summary['Occurrences'] / total_records) * 100

    return summary

def create_geographic_richness_summary(df):
    """
    Provides geographic summary and species richness statistics.
    
    - Geographic Summary: min, max, and mean for latitude and longitude.
    - Species Richness: For each checklist, the count of species with a nonzero observation,
      along with overall min, max, mean, and median richness.
    
    Assumes that the DataFrame has 'latitude' and 'longitude' columns among its first 11 fixed columns.
    """
    # Geographic summary for latitude and longitude
    geo_summary = {
        'Latitude': {
            'min': df['latitude'].min(),
            'max': df['latitude'].max(),
            'mean': df['latitude'].mean()
        },
        'Longitude': {
            'min': df['longitude'].min(),
            'max': df['longitude'].max(),
            'mean': df['longitude'].mean()
        }
    }
    
    # Calculate species richness per checklist (number of species with count > 0)
    species_columns = df.columns[11:]
    df['species_richness'] = (df[species_columns] > 0).sum(axis=1)
    
    richness_summary = {
    'min': int(df['species_richness'].min()),
    'max': int(df['species_richness'].max()),
    'mean': df['species_richness'].mean(),
    'median': int(df['species_richness'].median()),
    'deciles': {f"{int(q*100)}th": df['species_richness'].quantile(q) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
}
    
    geo_summary['Species Richness'] = richness_summary
    return geo_summary

def save_json(data, filepath):
    """Saves a dictionary as a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def cleanup_files(output_path, combined_filename, file_pattern=r"eBird_\d{4}_\d{2}\.csv"):
    """
    Removes individual CSV files from output_path that match the file_pattern.
    """
    files_to_remove = [f for f in os.listdir(output_path) if re.match(file_pattern, f)]
    for file in files_to_remove:
        os.remove(os.path.join(output_path, file))
    print(f"Removed {len(files_to_remove)} individual CSV files from {output_path}.")

def main():
    output_path = config.output_path
    combined_filename = config.output_filename

    # Combine the data
    final_df = combine_ebird_data(output_path)

    # Create summary statistics for species and geographic/richness data
    species_summary_df = create_summary_statistics(final_df)
    geo_richness_summary = create_geographic_richness_summary(final_df)

    # Convert species summary DataFrame to dictionary for JSON output
    species_summary = species_summary_df.to_dict(orient='index')

    # Save summaries as JSON files
    save_json(species_summary, os.path.join(output_path, "species_summary.json"))
    save_json(geo_richness_summary, os.path.join(output_path, "geo_richness_summary.json"))

    # Save the combined DataFrame as a single CSV file
    final_df.to_csv(os.path.join(output_path, combined_filename), index=False)

    # Clean up: Remove the individual CSV files from the output_path
    cleanup_files(output_path, combined_filename)

    # Output messages for the user
    print("Combined DataFrame saved as:", combined_filename)
    print("Species summary saved as: species_summary.json")
    print("Geographic and species richness summary saved as: geo_richness_summary.json")

if __name__ == "__main__":
    main()
