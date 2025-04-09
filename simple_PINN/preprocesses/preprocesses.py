import os
from simple_PINN.settings.config import ( 
    get_target_dir, get_log_path
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
    os.makedirs(get_target_dir(), exist_ok=True)
    # save config
    save_config.save_config(get_log_path())
    # fix random seed
    seed.torch_fix_seed()
