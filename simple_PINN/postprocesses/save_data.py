import numpy as np
from simple_PINN.settings import config

def save_data(t_grid, x_grid, u_pred, f_pred, history):
    """
    Save prediction results and training history to disk.

    This function stores:
    - Meshgrid arrays (t, x)
    - Predicted solution u(t, x)
    - Residual f(t, x)
    - Training loss history

    Files are saved to the directory specified by `config.get_target_dir()`.

    Parameters:
        t_grid (ndarray): Meshgrid of t-coordinates (2D).
        x_grid (ndarray): Meshgrid of x-coordinates (2D).
        u_pred (ndarray): Predicted solution on the grid (2D).
        f_pred (ndarray): Residual values on the grid (2D).
        history (ndarray): Training loss history, shape = [epochs, 5] with columns:
                           [epoch, loss_total, loss_u, loss_v, loss_pi]

    Returns:
        None
    """
    output_dir = config.get_target_dir()

    # Flatten and save the time and space grid
    np.savetxt(output_dir + "t_grid.dat", t_grid.flatten(), delimiter=",")
    np.savetxt(output_dir + "x_grid.dat", x_grid.flatten(), delimiter=",")

    # Save predicted solution and residual as 2D arrays
    np.savetxt(output_dir + "u_pred.dat", u_pred, delimiter=",")
    np.savetxt(output_dir + "f_pred.dat", f_pred, delimiter=",")

    # Save training loss history
    header = "epoch,loss_total,loss_u,loss_v,loss_pi"
    np.savetxt(
        output_dir + "loss_history.dat",
        history,
        fmt=["%d", "%.8e", "%.8e", "%.8e", "%.8e"],
        delimiter=",",
        header=header,
        comments=""
    )
