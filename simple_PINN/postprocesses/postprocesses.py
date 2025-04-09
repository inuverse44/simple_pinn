import numpy as np   
from simple_PINN.settings.config import ( 
    LOG_PATH
)
from simple_PINN.postprocesses import visualize
from simple_PINN.postprocesses import save_data
from simple_PINN.postprocesses import evaluate_error


def postprocess(pinn_model, history):
    """
    Postprocess the results.
    """
    # prediction
    n = 100
    t_pred = np.linspace(0, 1, n) # t=[0, 0.01, ..., 1]
    x_pred = np.linspace(-1, 1, n) # x=[-1, -0.98, ..., 1] predはpredictionの略
    t_grid, x_grid = np.meshgrid(t_pred, x_pred) # tとxのメッシュグリッドを作成
    X_in = np.block([[t_grid.flatten()], [x_grid.flatten()]]).T # tとxを結合

    # prediction
    u_pred, f_pred = pinn_model.predict(X_in)
    u_pred = u_pred.reshape(n, n)
    f_pred = f_pred.reshape(n, n)
    u_exact = np.sin(np.pi * x_grid) * np.cos(np.pi * t_grid)

    evaluate_error.evaluate_error(u_exact, u_pred, x_grid, t_grid, save_path=LOG_PATH)

    # save the prediction data
    save_data.save_data(t_grid, x_grid, u_pred, f_pred, history)
    # visualize the prediction
    visualize.plot_figures(pinn_model, history, t_pred, x_pred, u_pred, f_pred)
