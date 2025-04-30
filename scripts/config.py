from easydict import EasyDict as edict

config = edict()

config.seed = 42

# data params
config.data_region = 'STL'
config.train_path = "./data/gbif_full_no_overlap.csv" # "./data/gbif_full_filtered.csv" # "./data/gbif_full_filtered-train.csv" "./data/subsets/iNat_species_limited_max1000.csv"
config.val_path = None # "./data/gbif_full_filtered-validation.csv" # "./data/gbif_full_filtered-validation.csv" # None -- if approaching like memorization task.

# training params
config.modalities = 'loc'
config.loss_type = 'an_full' # max_entropy, an_full, an_full_weighted
config.pos_weight = 1024 # int or 'num_classes'
config.batch_size = 128 
config.max_epochs = 5
config.species_weights_method = 'uniform' # "uniform", "inversely_proportional_not_normalized", "inversely_proportional_sqrt", "inversely_proportional_clipped", "inversely_proportional"

config.lambda_1 = 1
config.lambda_2 = 0.8

# experiment params
if 'species_limited_max' in config.train_path:
    config.data_splits_naming = config.train_path.split('/')[-1].split('.')[0]  # Extract the last part of the path, assuming it's the name
else:
    config.data_splits_naming = 'train_val' if 'train' in config.train_path else 'full'

config.data_splits_naming = 'train_no_overlap'

weight_str = config.species_weights_method if config.species_weights_method != "uniform" else config.pos_weight

config.experiment_name = f'{config.data_region}-{config.data_splits_naming}-{config.modalities}-{config.loss_type}-{config.batch_size}-{weight_str}'

# testing
config.testing_df_path = "/data/cher/EcoBound/data/eBird/eBird_STL.csv"

# experiments

# Base
# loss_type = 'an_full' pos_weight = 1024 batch_size = 128

# How important is the pos_weighting? Very!
# loss_type = 'an_full' pos_weight = 1 batch_size = 128

# Do outputs change dramatically based on batch size?  Yes! Rare expecially.
# loss_type = 'an_full' pos_weight = 1024 batch_size = 1024

# Does ME look better? -- not with this weighting
# loss_type = 'max_entropy' pos_weight = 1 batch_size = 128

# Do we need a different weighting? (ME edition) This is not helpful. Wayyy too uncertain.
# loss_type = 'max_entropy' pos_weight = 1024 batch_size = 128

# Something in the middle? Time to test different ones out!
# loss_type = 'max_entropy' pos_weight = 128 batch_size = 128 

# Actually it needs to go up! (at least with normal weighting)
