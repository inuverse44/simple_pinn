import os
from simple_PINN.settings import config
from simple_PINN.settings import save_config
from simple_PINN.preprocesses import initialize
from simple_PINN.preprocesses import seed

def preprocesses():
    """
    Perform preprocessing before training the PINN model.

    This function handles:
    - Deleting or initializing the log file
    - Creating the output directory
    - Saving the current configuration to file
    - Fixing random seeds for reproducibility

    Returns:
        None
    """
    # Clear previous log file
    initialize.delete_log()

    # Ensure output directory exists
    os.makedirs(config.get_target_dir(), exist_ok=True)

    # Save current configuration to log
    save_config.save_config()

    # Fix random seed for reproducibility
    seed.torch_fix_seed()
