import numpy as np
from simple_PINN.settings.config import TARGET_DIR

def save_data(t_grid, x_grid, u_pred, f_pred, history):
    """
    データを保存する関数
    :param t_grid: メッシュグリッドのt座標
    :param x_grid: メッシュグリッドのx座標
    :param u_pred: 予測された物理量
    :param f_pred: 残差
    :param history: 学習履歴
    """
    # メッシュグリッド t, x（1次元配列）
    np.savetxt(TARGET_DIR + "t_grid.dat", t_grid.flatten(), delimiter=",")
    np.savetxt(TARGET_DIR + "x_grid.dat", x_grid.flatten(), delimiter=",")

    # 予測された物理量 u(t, x), 残差 f(t, x)（2Dグリッドとして保存）
    np.savetxt(TARGET_DIR + "u_pred.dat", u_pred, delimiter=",")
    np.savetxt(TARGET_DIR + "f_pred.dat", f_pred, delimiter=",")

    # 学習履歴（epochは整数、他は浮動小数で保存）
    header = "epoch,loss_total,loss_u,loss_v,loss_pi"
    np.savetxt(
        TARGET_DIR + "loss_history.dat",
        history,
        fmt=["%d", "%.8e", "%.8e", "%.8e", "%.8e"],  # epochは整数、それ以外は指数表記
        delimiter=",",
        header=header,
        comments=""
    )