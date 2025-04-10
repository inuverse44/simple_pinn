import numpy as np
from simple_PINN.settings import config
from simple_PINN.postprocesses import visualize
from simple_PINN.postprocesses import save_data
from simple_PINN.postprocesses import evaluate_error

def postprocess(pinn_model, history):
    """
    Postprocess the results after training the PINN model.

    This function performs:
    - Prediction of u and residual f on a 2D grid
    - Calculation and logging of L1, L2, and max error norms
    - Saving prediction and residual data to disk
    - Generating visualizations (sampling, prediction, residual, etc.)

    Parameters:
        pinn_model (PINN): Trained PINN model.
        history (ndarray): Training history with losses (shape: [n_epochs, 5]).

    Returns:
        None
    """

    # Create prediction grid
    n = 100
    t_pred = np.linspace(0, 1, n)
    x_pred = np.linspace(-1, 1, n)
    t_grid, x_grid = np.meshgrid(t_pred, x_pred)
    X_in = np.block([[t_grid.flatten()], [x_grid.flatten()]]).T

    # Perform model prediction
    u_pred, f_pred = pinn_model.predict(X_in)
    u_pred = u_pred.reshape(n, n)
    f_pred = f_pred.reshape(n, n)

    # Compute exact solution for evaluation
    u_exact = np.sin(np.pi * x_grid) * np.cos(np.pi * t_grid)

    # Evaluate and log errors
    evaluate_error.evaluate_error(
        u_exact, u_pred, x_grid, t_grid,
        save_path=config.get_log_path()
    )

    # Save prediction and residual data
    save_data.save_data(t_grid, x_grid, u_pred, f_pred, history)

    # Generate visualizations
    visualize.plot_figures(pinn_model, history, t_pred, x_pred, u_pred, f_pred)
