# EcoBound
Active Learning-Based Species Distribution Estimation Under Resource Constraint

## Preparing Your Environment

Run the following to create the environment:
```
conda env create --file environment.yml
```

Then activate the environment:
```
conda activate ecobound
```

If you make changes to the environment to get it working you can do
the following to update:
```
conda env update --file environment.yml --prune
```

## Running scripts

All scripts are run from the EcoBound level:
```
cd EcoBound
```

Generating priors. All outputs will be in outputs folder:
```
# Train the model
python scripts/model.py

# Run inference on all species
python scripts/inference.py
```

Maps can also be created for specific species:
```
python scripts/plot_map.py
```