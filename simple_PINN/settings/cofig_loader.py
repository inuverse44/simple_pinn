import yaml

def load_configs(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)['configs']
