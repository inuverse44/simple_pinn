import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from simple_PINN.settings import config

def sampling_points():
    """
    Visualize sampling points for initial, boundary, and collocation conditions.

    Returns:
        None
    """
    t_initial = config.get("t_initial")
    x_initial = config.get("x_initial")
    u_initial = config.get("u_initial")
    t_boundary = config.get("t_boundary")
    x_boundary = config.get("x_boundary")
    u_boundary = config.get("u_boundary")
    t_region = config.get("t_region")
    x_region = config.get("x_region")

    plt.figure(figsize=(10, 2))
    plt.scatter(t_initial, x_initial, c=u_initial)
    plt.scatter(t_boundary, x_boundary, c=u_boundary)
    plt.colorbar()
    plt.scatter(t_region, x_region, c="gray", s=5, marker="x")

    plt.title('Sampling Points')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')

    path = config.get_target_dir() + "sampling_points.pdf"
    plt.savefig(path, bbox_inches='tight')


def initial_conditions():
    """
    Visualize the initial condition u(x, 0).

    Returns:
        None
    """
    x_initial = config.get("x_initial")
    u_initial = config.get("u_initial")

    plt.figure(figsize=(3, 2))
    plt.scatter(x_initial, u_initial)

    plt.title('Initial Condition')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')

    path = config.get_target_dir() + "initial_conditions.pdf"
    plt.savefig(path, bbox_inches='tight')


def loss(history):
    """
    Visualize loss history during training.

    Parameters:
        history (ndarray): Training loss history [epoch, loss_total, loss_u, loss_v, loss_pi]

    Returns:
        None
    """
    plt.figure(figsize=(10, 2))
    plt.plot(history[:, 0], history[:, 1], label='loss_total')
    plt.plot(history[:, 0], history[:, 2], label='loss_u')
    plt.plot(history[:, 0], history[:, 3], label='loss_v')
    plt.plot(history[:, 0], history[:, 4], label='loss_pi')

    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss History')
    plt.legend()

    path = config.get_target_dir() + "loss_history.pdf"
    plt.savefig(path, bbox_inches='tight')


def prediction(t_pred, x_pred, u_pred):
    """
    Visualize predicted solution u(t, x).

    Parameters:
        t_pred (ndarray): 1D array of time points.
        x_pred (ndarray): 1D array of spatial points.
        u_pred (ndarray): 2D predicted solution grid.

    Returns:
        None
    """
    plt.figure(figsize=(10, 2))
    plt.contourf(t_pred, x_pred, u_pred, 32)
    plt.colorbar()

    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.title(r'Predicted $u(t, x)$')

    path = config.get_target_dir() + "prediction.pdf"
    plt.savefig(path, bbox_inches='tight')


def residual(t_pred, x_pred, f_pred):
    """
    Visualize the residual f(t, x) from the governing equation.

    Parameters:
        t_pred (ndarray): 1D array of time points.
        x_pred (ndarray): 1D array of spatial points.
        f_pred (ndarray): 2D residual grid.

    Returns:
        None
    """
    plt.figure(figsize=(10, 2))
    plt.contourf(t_pred, x_pred, f_pred, 32)
    plt.colorbar()

    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.title(r'Residual $f(t, x)$')

    path = config.get_target_dir() + "residual.pdf"
    plt.savefig(path, bbox_inches='tight')


def time_evolution(pinn_model, x_pred):
    """
    Plot time evolution of u(x, t) and f(x, t) at selected time snapshots.

    Parameters:
        pinn_model (PINN): Trained PINN model.
        x_pred (ndarray): 1D array of spatial points.

    Returns:
        None
    """
    times = [0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(times), figsize=(15, 2), sharey=True)

    for i, t in enumerate(times):
        t_pred = np.ones([100, 1]) * t
        X_in = np.column_stack([t_pred, x_pred])
        u_pred, f_pred = pinn_model.predict(X_in)
        u_exact = np.sin(np.pi * x_pred.flatten()) * np.cos(np.pi * t)

        ax = axes[i]
        ax.plot(x_pred, u_exact, label=r'$u_{\rm exact}$', linewidth=1, linestyle='dashed', color='black')
        ax.plot(x_pred, u_pred, label=r'$u_{\rm pred}$', linewidth=0.5)
        ax.plot(x_pred, f_pred, label=r'$f_{\rm pred}$', linewidth=0.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(f"t = {t:.2f}")

        if i == 0:
            ax.set_ylabel("u")

    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()

    path = config.get_target_dir() + "time_evolution.pdf"
    plt.savefig(path, bbox_inches='tight')


def difference(pinn_model, x_pred):
    """
    Visualize the difference u_exact - u_pred over time.

    Parameters:
        pinn_model (PINN): Trained PINN model.
        x_pred (ndarray): 1D array of spatial points.

    Returns:
        None
    """
    times = [0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(times), figsize=(15, 2), sharey=True)

    for i, t in enumerate(times):
        t_pred = np.ones([100, 1]) * t
        X_in = np.column_stack([t_pred, x_pred])
        u_pred, _ = pinn_model.predict(X_in)
        u_exact = np.sin(np.pi * x_pred.flatten()) * np.cos(np.pi * t)
        diff_u = u_pred.flatten() - u_exact
        diff_u2 = -diff_u

        ax = axes[i]
        ax.plot(x_pred, diff_u, label=r'$u_{\rm exact} - u_{\rm pred}$',
                linewidth=1, linestyle='solid', color='black')
        ax.plot(x_pred, diff_u2, linewidth=1, linestyle='dashed', color='black')
        ax.set_ylim(0, 1.5)
        ax.set_title(f"t = {t:.2f}")

        if i == 0:
            ax.set_ylabel("difference")

    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=1, bbox_to_anchor=(0.5, -0.15))

    fig.tight_layout()
    path = config.get_target_dir() + "difference.pdf"
    plt.savefig(path, bbox_inches='tight')


def plot_figures(pinn_model, history, t_pred, x_pred, u_pred, f_pred):
    """
    Generate and save all relevant figures.

    Parameters:
        pinn_model (PINN): Trained PINN model.
        history (ndarray): Training loss history.
        t_pred (ndarray): 1D array of time points.
        x_pred (ndarray): 1D array of spatial points.
        u_pred (ndarray): 2D predicted solution.
        f_pred (ndarray): 2D residual.

    Returns:
        None
    """
    sampling_points()
    initial_conditions()
    loss(history)
    prediction(t_pred, x_pred, u_pred)
    residual(t_pred, x_pred, f_pred)
    time_evolution(pinn_model, x_pred)
    difference(pinn_model, x_pred)
