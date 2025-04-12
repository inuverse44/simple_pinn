import os
import yaml
from itertools import product

# ==== CONFIGURATION ====
N_INITIAL = 100
N_BOUNDARY = 100
N_REGION = 5000
VELOCITY = 1

epoch_options = [100, 500, 1000, 2000, 3000, 4000, 5000]
lr_options = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
weight_options = [1e-3, 5e-3, 1e-2, 1e-2, 1e-1, 5e-1, 1.0]

output_root = "output"
config_yaml_path = "config.yaml"
todo_yaml_path = "todo.yaml"

# ==== HELPERS ====

def make_folder_name(n_init, n_boun, n_regi, maxep, lr, w, v):
    return f"init={n_init}_boun={n_boun}_regi={n_regi}_maxep={maxep}_lr={lr:.4f}_w={w:.4f}_v={v}"

def is_already_done(folder_path):
    """Check if log.txt exists in the output folder."""
    return os.path.isfile(os.path.join(folder_path, "log.txt"))

def make_config_entry(name, maxep, lr, w):
    return {
        "name": name,
        "N_INITIAL": N_INITIAL,
        "N_BOUNDARY": N_BOUNDARY,
        "N_REGION": N_REGION,
        "MAX_EPOCHS_FOR_MODEL": maxep,
        "MAX_EPOCHS_FOR_FITTING": 1000,
        "LEARNING_RATE": lr,
        "PI_WEIGHT": w,
        "VELOCITY": VELOCITY,
        "EPOCH_SEPARATOR": 10
    }

def load_existing_configs(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = yaml.safe_load(f)
            return existing.get("configs", [])
    return []

def save_yaml(path, configs):
    with open(path, "w") as f:
        yaml.dump({"configs": configs}, f, sort_keys=False)

# ==== MAIN ====

# Step 1: Generate all combinations
all_combinations = product(epoch_options, lr_options, weight_options)

# Step 2: Load existing config.yaml
existing_configs = load_existing_configs(config_yaml_path)
existing_names = {cfg["name"] for cfg in existing_configs}

# Step 3: Filter out existing and completed configs
new_configs = []

for maxep, lr, w in all_combinations:
    folder_name = make_folder_name(N_INITIAL, N_BOUNDARY, N_REGION, maxep, lr, w, VELOCITY)
    folder_path = os.path.join(output_root, folder_name)
    config_name = f"maxep{maxep}_lr{lr:.0e}_w{w:.0e}"

    if config_name in existing_names:
        print(f"⏭ Already in config.yaml: {config_name}")
        continue

    if is_already_done(folder_path):
        print(f"✔ Already completed: {folder_name}")
        continue

    new_configs.append(make_config_entry(config_name, maxep, lr, w))

# Step 4: Save all configs (config.yaml) and only-new (todo.yaml)
if new_configs:
    full_configs = existing_configs + new_configs
    save_yaml(config_yaml_path, full_configs)
    save_yaml(todo_yaml_path, new_configs)
    print(f"\n✅ Appended {len(new_configs)} new configs to: {config_yaml_path}")
    print(f"📝 Saved {len(new_configs)} TODO configs to: {todo_yaml_path}")
else:
    print("\n✅ No new configs to append. Everything is up to date.")
