import os

from simple_PINN.settings.config import ( 
    get_target_dir, get_log_path,
    N_INITIAL, N_BOUNDARY, N_REGION,
    LEARNING_RATE, PI_WEIGHT, VELOCITY,
    MAX_EPOCHS_FOR_MODEL, MAX_EPOCHS_FOR_FITTING
)

def save_config(save_path=get_log_path()):
    """
    設定を保存
    @param save_path: 保存先のパス
    """
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "a") as f:
            f.write("=== Settings ===\n")
            f.write(f"Initial points  : {N_INITIAL}\n")
            f.write(f"Boundary points : {N_BOUNDARY}\n")
            f.write(f"Region points   : {N_REGION}\n")
            f.write(f"Max epochs      : {MAX_EPOCHS_FOR_MODEL}\n")
            f.write(f"Max epochs (fit): {MAX_EPOCHS_FOR_FITTING}\n")
            f.write(f"Learning rate   : {LEARNING_RATE}\n")
            f.write(f"PI weight       : {PI_WEIGHT}\n")
            f.write(f"Velocity        : {VELOCITY}\n")
            f.write("\n")
