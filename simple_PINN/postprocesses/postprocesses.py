import numpy as np
from simple_PINN.settings import config  # ← 追加
from simple_PINN.postprocesses import visualize
from simple_PINN.postprocesses import save_data
from simple_PINN.postprocesses import evaluate_error

def postprocess(pinn_model, history):
    """
    Postprocess the results.
    """
    # prediction grid
    n = 100
    t_pred = np.linspace(0, 1, n)
    x_pred = np.linspace(-1, 1, n)
    t_grid, x_grid = np.meshgrid(t_pred, x_pred)
    X_in = np.block([[t_grid.flatten()], [x_grid.flatten()]]).T

    # model prediction
    u_pred, f_pred = pinn_model.predict(X_in)
    u_pred = u_pred.reshape(n, n)
    f_pred = f_pred.reshape(n, n)
    u_exact = np.sin(np.pi * x_grid) * np.cos(np.pi * t_grid)

    # 評価指標を計算＆ログ出力
    evaluate_error.evaluate_error(
        u_exact, u_pred, x_grid, t_grid,
        save_path=config.get_log_path()
    )

    # 結果を保存
    save_data.save_data(t_grid, x_grid, u_pred, f_pred, history)

    # 図示
    visualize.plot_figures(pinn_model, history, t_pred, x_pred, u_pred, f_pred)
