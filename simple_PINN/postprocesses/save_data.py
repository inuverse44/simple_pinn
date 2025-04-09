import numpy as np
from simple_PINN.settings import config

def save_data(t_grid, x_grid, u_pred, f_pred, history):
    """
    データを保存する関数
    :param t_grid: メッシュグリッドのt座標
    :param x_grid: メッシュグリッドのx座標
    :param u_pred: 予測された物理量
    :param f_pred: 残差
    :param history: 学習履歴
    """
    output_dir = config.get_target_dir()

    # メッシュグリッド t, x（1次元配列）
    np.savetxt(output_dir + "t_grid.dat", t_grid.flatten(), delimiter=",")
    np.savetxt(output_dir + "x_grid.dat", x_grid.flatten(), delimiter=",")

    # 予測された物理量 u(t, x), 残差 f(t, x)
    np.savetxt(output_dir + "u_pred.dat", u_pred, delimiter=",")
    np.savetxt(output_dir + "f_pred.dat", f_pred, delimiter=",")

    # 学習履歴
    header = "epoch,loss_total,loss_u,loss_v,loss_pi"
    np.savetxt(
        output_dir + "loss_history.dat",
        history,
        fmt=["%d", "%.8e", "%.8e", "%.8e", "%.8e"],
        delimiter=",",
        header=header,
        comments=""
    )
