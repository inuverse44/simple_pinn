import numpy as np
from simple_PINN.training.NN import NN
from simple_PINN.training.PINN import PINN
from simple_PINN.settings import config  # ← 関数ベース構成用

def training():
    # データの取得（configから）
    t_initial = config.get("t_initial")
    x_initial = config.get("x_initial")
    u_initial = config.get("u_initial")
    v_initial = config.get("v_initial")

    t_boundary = config.get("t_boundary")
    x_boundary = config.get("x_boundary")
    u_boundary = config.get("u_boundary")

    t_region = config.get("t_region")
    x_region = config.get("x_region")

    max_epochs = config.get("MAX_EPOCHS_FOR_MODEL")

    # 学習用のデータを作成
    X_bc = np.block([[t_initial, t_boundary], [x_initial, x_boundary]]).T
    Y_bc = np.block([[u_initial, u_boundary]]).T
    V_ic = v_initial.reshape(-1, 1)
    X_region = np.block([[t_region], [x_region]]).T

    model = NN(2, 1)
    pinn_model = PINN(model)

    history = pinn_model.fit(X_bc, Y_bc, X_region, v_ic=V_ic, max_epochs=max_epochs)
    return pinn_model, history
