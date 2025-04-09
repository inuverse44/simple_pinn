
import time
from simple_PINN.preprocesses import preprocesses
from simple_PINN.training import training
from simple_PINN.postprocesses import postprocesses

def main_PINN():
    start_time = time.time()

    # SETTINGS (initialization, seed setting, etc.)
    preprocesses.preprocesses()

    # TRAINING (PINN)
    pinn_model, history = training.training()

    # POSTPROCESSING (visualization, save data, etc.)
    postprocesses.postprocess(pinn_model, history)
    
    
    end_time = time.time()
    from simple_PINN.settings.config import LOG_PATH
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    if LOG_PATH is not None:
        with open(LOG_PATH, "a") as f:
            f.write("=== Execution time ===\n")
            f.write(f"Total execution time: {end_time - start_time:.2f} seconds\n")
            f.write("\n")


  