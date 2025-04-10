import os
from simple_PINN.settings import config

def save_config(save_path=None):
    """
    Save the current experiment configuration to a log file.

    The function writes all key hyperparameters and dataset sizes
    into the specified log file. If no path is provided, the default
    path from config.get_log_path() is used.

    Parameters:
        save_path (str, optional): Path to the log file. Defaults to config.get_log_path().

    Returns:
        None
    """
    if save_path is None:
        save_path = config.get_log_path()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "a") as f:
        f.write("=== Settings ===\n")
        f.write(f"Initial points  : {config.get('N_INITIAL')}\n")
        f.write(f"Boundary points : {config.get('N_BOUNDARY')}\n")
        f.write(f"Region points   : {config.get('N_REGION')}\n")
        f.write(f"Max epochs      : {config.get('MAX_EPOCHS_FOR_MODEL')}\n")
        f.write(f"Max epochs (fit): {config.get('MAX_EPOCHS_FOR_FITTING')}\n")
        f.write(f"Learning rate   : {config.get('LEARNING_RATE')}\n")
        f.write(f"PI weight       : {config.get('PI_WEIGHT')}\n")
        f.write(f"Velocity        : {config.get('VELOCITY')}\n")
        f.write("\n")
