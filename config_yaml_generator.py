import yaml

# Define parameter search space
epoch_options = [100, 1000, 3000, 5000]           # Max training epochs
lr_options = [1e-4, 1e-3, 1e-2]                   # Learning rates
weight_options = [1e-3, 1e-2, 1e-1, 1.0]          # Residual (physics) loss weights

configs = []

# Generate all combinations of (epoch, learning rate, PI weight)
for maxep in epoch_options:
    for lr in lr_options:
        for w in weight_options:
            config = {
                "name": f"maxep{maxep}_lr{lr:.0e}_w{w:.0e}",
                "N_INITIAL": 100,
                "N_BOUNDARY": 100,
                "N_REGION": 5000,
                "MAX_EPOCHS_FOR_MODEL": maxep,
                "MAX_EPOCHS_FOR_FITTING": 1000,
                "LEARNING_RATE": lr,
                "PI_WEIGHT": w,
                "VELOCITY": 1,
                "EPOCH_SEPARATOR": 10
            }
            configs.append(config)

# Wrap in top-level "configs" key for compatibility with load_configs()
config_dict = {"configs": configs}

# Write to config.yaml
output_path = "config.yaml"
with open(output_path, "w") as f:
    yaml.dump(config_dict, f, sort_keys=False)

print(f"Configuration saved to: {output_path}")
