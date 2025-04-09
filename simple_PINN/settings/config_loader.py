import yaml

def load_configs(yaml_path):
    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data["configs"]
