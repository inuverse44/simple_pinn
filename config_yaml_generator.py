import yaml

# 設定生成範囲
epoch_options = [100, 1000, 3000, 5000]
lr_options = [1e-4, 1e-3, 1e-2]
weight_options = [1e-3, 1e-2, 1e-1, 1.0]

configs = []
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

config_dict = {"configs": configs}

# 保存
output_path = "config.yaml"
with open(output_path, "w") as f:
    yaml.dump(config_dict, f, sort_keys=False)

output_path
