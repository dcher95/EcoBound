import tarfile
import gzip
import logging
from datetime import datetime
from tqdm import tqdm
from typing import Tuple
import os

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def filter_and_save_data(
    tar_path: str,
    csv_filename: str,
    output_filename: str,
    bounding_box: Tuple[float, float, float, float],
    date_range: Tuple[datetime, datetime]
) -> None:
    """
    Extracts, filters, and saves relevant rows from a large gzipped CSV inside a tar archive.
    
    Args:
        tar_path (str): Path to the tar archive.
        csv_filename (str): Name of the gzipped CSV file inside the archive.
        output_filename (str): Output CSV file to save filtered rows.
        bounding_box (Tuple[float, float, float, float]): (lon_min, lon_max, lat_min, lat_max).
        date_range (Tuple[datetime, datetime]): (start_date, end_date).
    """
    # Check if output file already exists. If so, skip processing.
    if os.path.exists(output_filename):
        logging.info(f"File {output_filename} already exists. Skipping processing.")
        return

    # Ensure output directory exists before writing
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    lon_min, lon_max, lat_min, lat_max = bounding_box
    start_date, end_date = date_range

    try:
        with tarfile.open(tar_path, "r") as tar:
            gz_file = tar.extractfile(csv_filename)
            if gz_file is None:
                raise FileNotFoundError(f"{csv_filename} not found in the tar archive.")

            with gzip.open(gz_file, mode='rt') as gz, open(output_filename, mode='w', encoding="ascii", errors="ignore") as out_txt:
                # Read first line as header
                header = gz.readline().strip().split("\t")  # Adjust delimiter if needed
                out_txt.write("\t".join(header) + "\n")  # Write header

                # Get column indices for required fields
                try:
                    lat_idx = header.index("LATITUDE")
                    lon_idx = header.index("LONGITUDE")
                    date_idx = header.index("OBSERVATION DATE")
                except ValueError as e:
                    logging.error(f"Missing expected column in the file header: {e}")
                    exit(1)

                # Process rows
                for line in tqdm(gz, desc="Processing rows", unit="rows"):
                    try:
                        row = line.strip().split("\t")  # Adjust delimiter if needed
                        lat, lon = float(row[lat_idx]), float(row[lon_idx])
                        obs_date = datetime.strptime(row[date_idx], "%Y-%m-%d")

                        # Apply filters
                        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max and start_date <= obs_date <= end_date:
                            out_txt.write("\t".join(row) + "\n")  # Write filtered row
                    except (ValueError, IndexError) as e:
                        logging.warning(f"Skipping invalid row: {row} - Error: {e}")
                        continue

            logging.info(f"Filtered dataset saved as: {output_filename}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

def main() -> None:
    """
    Main function to define parameters and run the data processing pipeline.
    """
    tar_path_sampling = config.tar_path_sampling
    csv_filename_sampling = config.csv_filename_sampling
    output_filename_sampling = f"{config.output_path}stl_ebird_sampling_data.txt"

    tar_path_observations = config.tar_path_observations
    csv_filename_observations = config.csv_filename_observations
    output_filename_observations = f"{config.output_path}stl_ebird_observations_data.txt"

    # Define boundaries for area (St. Louis region example)
    bounding_box = config.bounding_box
    
    # Define date range for filtering (2010 - 2025)
    # date_range = (datetime(2010, 1, 1), datetime(2025, 12, 31))
    date_range = (
        datetime.strptime(config.beginning_date, "%Y-%m-%d"),
        datetime.strptime(config.ending_date, "%Y-%m-%d")
    )

    # Process and filter data (for sampling)
    filter_and_save_data(tar_path_sampling, csv_filename_sampling, output_filename_sampling, bounding_box, date_range)

    # Process and filter data (for observations)
    filter_and_save_data(tar_path_observations, csv_filename_observations, output_filename_observations, bounding_box, date_range)

if __name__ == "__main__":
    main()
