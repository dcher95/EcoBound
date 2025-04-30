from easydict import EasyDict as edict

config = edict()

config.seed=42

# Coordinates
config.start_lat = 38.648974
config.start_lon = -90.310818

# Sampling parameters
config.degrees_to_km_conversion = 100
config.seconds_per_km = 90
config.seconds_to_sample = 600
config.lambda_param = 0.02
config.seconds_per_degree = config.seconds_per_km * config.degrees_to_km_conversion
config.seconds_per_trip = 4 * 3600
config.avail_trips = 50

# Testing species
config.species_to_plot = [
    "Agelaius phoeniceus", 'Baeolophus bicolor', "Cardinalis cardinalis", "Sialia sialis",  
    "Ardea herodias"
]
