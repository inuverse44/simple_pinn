from scipy.integrate import simpson
import numpy as np
import os
from simple_PINN.settings import config  

def evaluate_error(u_exact, u_pred, x_grid, t_grid, save_path=None):
    """
    Evaluate prediction error using L1, L2, and L-infinity norms.

    Parameters:
        u_exact (ndarray): Exact solution on the grid.
        u_pred (ndarray): Predicted solution on the grid.
        x_grid (ndarray): Meshgrid of x-coordinates.
        t_grid (ndarray): Meshgrid of t-coordinates.
        save_path (str, optional): Path to save the error metrics. Defaults to config.get_log_path().

    Returns:
        tuple: (L1 norm, L2 norm, Max norm)
    """

    # Compute absolute and squared differences
    abs_diff = np.abs(u_exact - u_pred)
    sq_diff = (u_exact - u_pred)**2

    # Grid spacing
    dx = x_grid[1, 0] - x_grid[0, 0]  # spacing in x-direction
    dt = t_grid[0, 1] - t_grid[0, 0]  # spacing in t-direction

    # L1 norm (2D integral using Simpson's rule)
    l1_error = simpson(simpson(abs_diff, dx=dt, axis=1), dx=dx, axis=0)

    # L2 norm (square root of 2D integral of squared error)
    l2_squared = simpson(simpson(sq_diff, dx=dt, axis=1), dx=dx, axis=0)
    l2_error = np.sqrt(l2_squared)

    # L-infinity norm (max absolute error)
    linf_error = np.max(abs_diff)

    # Print results
    print(f"L1 norm    : {l1_error:.5e}")
    print(f"L2 norm    : {l2_error:.5e}")
    print(f"Max error  : {linf_error:.5e}")

    # Determine save path if not specified
    if save_path is None:
        save_path = config.get_log_path()

    # Save metrics to file
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "a") as f:
            f.write("=== Evaluation Metrics ===\n")
            f.write(f"L1 norm    : {l1_error:.5e}\n")
            f.write(f"L2 norm    : {l2_error:.5e}\n")
            f.write(f"Max error  : {linf_error:.5e}\n")
            f.write("\n")

    return l1_error, l2_error, linf_error
