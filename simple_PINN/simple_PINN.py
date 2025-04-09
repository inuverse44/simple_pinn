import time
from simple_PINN.preprocesses import preprocesses
from simple_PINN.training import training
from simple_PINN.postprocesses import postprocesses
from simple_PINN.settings import config  # ← 修正！

def main_PINN():
    start_time = time.time()

    # SETTINGS (initialization, seed setting, etc.)
    preprocesses.preprocesses()

    # TRAINING (PINN)
    pinn_model, history = training.training()

    # POSTPROCESSING (visualization, save data, etc.)
    postprocesses.postprocess(pinn_model, history)

    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")

    log_path = config.get_log_path()
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write("=== Execution time ===\n")
            f.write(f"Total execution time: {end_time - start_time:.2f} seconds\n")
            f.write("\n")
