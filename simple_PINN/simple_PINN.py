import numpy as np
from simple_PINN.NN import NN
from simple_PINN.PINN import PINN
from simple_PINN.preprocesses import torch_fix_seed
from simple_PINN.settings import (
    t_initial, x_initial, u_initial, v_initial, 
    t_boundary, x_boundary, u_boundary,
    t_region, x_region, 
    MAX_EPOCHS_FOR_MODEL
)
from simple_PINN import visualize

def main_PINN():
    ###############
    # 問題設定
    ###############

    # 乱数のシードを固定
    torch_fix_seed()

    # 学習用のデータ
    # 注意：初期条件（t=0）と境界条件（t∈[0,1]）を合成してX_bcにしているため、tやuも「初期条件+境界条件=200個」になっている
    X_bc = np.block([[t_initial, t_boundary], [x_initial, x_boundary]]).T #x軸のboundary condition
    Y_bc = np.block([[u_initial, u_boundary]]).T #y軸のboundary condition
    V_ic = v_initial.reshape(-1, 1) # 初期条件の速度(t=0)
    X_region = np.block([[t_region], [x_region]]).T 

    ###############
    # 学習
    ###############
    model = NN(2, 1)
    pinn_model = PINN(model)

    # モデルを学習
    history = pinn_model.fit(X_bc, Y_bc, X_region, v_ic=V_ic, max_epochs=MAX_EPOCHS_FOR_MODEL)

    # 予測したいポイント
    n = 100
    t_pred = np.linspace(0, 1, n) # t=[0, 0.01, ..., 1]
    x_pred = np.linspace(-1, 1, n) # x=[-1, -0.98, ..., 1] predはpredictionの略
    t_grid, x_grid = np.meshgrid(t_pred, x_pred) # tとxのメッシュグリッドを作成
    X_in = np.block([[t_grid.flatten()], [x_grid.flatten()]]).T # tとxを結合

    # 予測 (prediction)
    u_pred, f_pred = pinn_model.predict(X_in)
    u_pred = u_pred.reshape(n, n)
    f_pred = f_pred.reshape(n, n)


    ###############
    # 可視化
    ###############
    # サンプリングした点を可視化
    visualize.sampling_points()
    # 初期条件を可視化
    visualize.initial_conditions()
    # lossのh履歴を可視化
    visualize.loss(history)
    # 物理量を可視化
    visualize.prediction(t_pred, x_pred, u_pred)
    # 基礎方程式の残差を可視化
    visualize.residual(t_pred, x_pred, f_pred)
    # 物理量の時間変化を可視化
    visualize.time_evolution(pinn_model, x_pred)