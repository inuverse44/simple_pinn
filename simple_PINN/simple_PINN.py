import time
from simple_PINN.preprocesses import preprocesses
from simple_PINN.training import training
from simple_PINN.postprocesses import postprocesses
from simple_PINN.settings import config

def main_PINN():
    """
    Main function to execute one full PINN training run.

    This function orchestrates:
    - Preprocessing: log initialization, seed fixing, config saving
    - Training: PINN model training with initial/boundary/collocation data
    - Postprocessing: prediction, evaluation, visualization, and saving

    Logs the total execution time to the configured log file.

    Returns:
        None
    """
    start_time = time.time()

    # Preprocessing (log file, seed, config, directory creation)
    preprocesses.preprocesses()

    # Model training
    pinn_model, history = training.training()

    # Postprocessing (evaluation, saving, visualization)
    postprocesses.postprocess(pinn_model, history)

    end_time = time.time()

    # Console output
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    # Save timing info to log
    log_path = config.get_log_path()
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write("=== Execution time ===\n")
            f.write(f"Total execution time: {end_time - start_time:.2f} seconds\n")
            f.write("\n")
