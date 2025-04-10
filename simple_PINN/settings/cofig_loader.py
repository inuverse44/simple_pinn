import yaml

def load_configs(yaml_path):
    """
    Load experiment configurations from a YAML file.

    The YAML file should contain a top-level key "configs"
    with a list of configuration dictionaries.

    Parameters:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        list[dict]: List of experiment configuration dictionaries.
    """
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)['configs']
