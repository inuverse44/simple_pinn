import numpy as np

# Global configuration dictionary (populated via apply_config)
_config = {}

def apply_config(cfg):
    """
    Apply a configuration dictionary and generate additional derived parameters.

    This function:
    - Copies user-defined values from the config dictionary
    - Computes sampling points (initial, boundary, interior)
    - Constructs output and log paths based on parameter values

    Parameters:
        cfg (dict): Configuration dictionary from YAML.

    Returns:
        None
    """
    global _config
    _config = cfg.copy()

    # Initial condition points (t = 0)
    _config['t_initial'] = np.zeros(cfg['N_INITIAL'])
    _config['x_initial'] = np.linspace(-1, 1, cfg['N_INITIAL'])
    _config['u_initial'] = np.sin(np.pi * _config['x_initial'])
    _config['v_initial'] = np.zeros_like(_config['x_initial'])

    # Boundary condition points (x = ±1)
    _config['t_boundary'] = np.random.rand(cfg['N_BOUNDARY'])
    _config['x_boundary'] = np.random.choice([-1, 1], cfg['N_BOUNDARY'])
    _config['u_boundary'] = np.zeros(cfg['N_BOUNDARY'])

    # Collocation (interior) points
    _config['t_region'] = np.random.rand(cfg['N_REGION'])
    _config['x_region'] = np.random.uniform(-1, 1, cfg['N_REGION'])

    # Output and logging paths
    _config['OUTPUT_DIR'] = "output/"
    _config['TARGET_DIR'] = _config['OUTPUT_DIR'] + \
        f"init={cfg['N_INITIAL']}_boun={cfg['N_BOUNDARY']}_regi={cfg['N_REGION']}_maxep={cfg['MAX_EPOCHS_FOR_MODEL']}_lr={cfg['LEARNING_RATE']}_w={cfg['PI_WEIGHT']}_v={cfg['VELOCITY']}/"
    _config['LOG_PATH'] = _config['TARGET_DIR'] + "log.txt"


def get(key):
    """
    Retrieve a value from the global configuration dictionary.

    Parameters:
        key (str): Key to retrieve.

    Returns:
        Any: Corresponding value.
    """
    return _config[key]


def get_target_dir():
    """
    Return the output directory path for saving experiment results.

    Returns:
        str: Output directory path.
    """
    return _config['TARGET_DIR']


def get_log_path():
    """
    Return the path to the log file.

    Returns:
        str: Log file path.
    """
    return _config['LOG_PATH']
