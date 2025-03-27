from easydict import EasyDict as edict

config = edict()

config.tar_path_sampling = "/data/cher/user_cher/eBird/ebd_sampling_relFeb-2025.tar"
config.csv_filename_sampling = "ebd_sampling_relFeb-2025.txt.gz"

config.tar_path_observations = "/data/cher/user_cher/eBird/ebd_relFeb-2025.tar"
config.csv_filename_observations = "ebd_relFeb-2025.txt.gz"

config.output_path = "/data/cher/EcoBound/data/eBird/"
config.output_filename = "eBird_STL.csv"

config.bounding_box = (-90.6809899999999942, -90.0909899999996924, 38.4560099999999991, 38.8860099999999136)

config.beginning_date = "2010-1-1"
config.ending_date = "2025-12-31"