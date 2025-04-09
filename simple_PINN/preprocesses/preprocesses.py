import os
from simple_PINN.settings.config import ( 
    TARGET_DIR, LOG_PATH
)
from simple_PINN.settings import save_config
from simple_PINN.preprocesses import initialize
from simple_PINN.preprocesses import seed


def preprocesses():
    """
    Preprocesses the data.
    """
    # delete log file
    initialize.delete_log()

    # setting path
    os.makedirs(TARGET_DIR, exist_ok=True)
    # save config
    save_config.save_config(LOG_PATH)
    # fix random seed
    seed.torch_fix_seed()
