import numpy as np
import matplotlib.pyplot as plt
from pinn_burgers.nn import NN
from pinn_burgers.burgers_nn import BurgersPINN
from pinn_burgers.preprocesses import torch_fix_seed
from pinn_burgers.settings import (
    t_initial, x_initial, u_initial,
    t_boundary, x_boundary, u_boundary,
    t_region, x_region, 
    MAX_EPOCHS_FOR_MODEL
)

def main():
    # 乱数のシードを固定
    torch_fix_seed()

    ###### 問題設定の可視化 ######
    # サンプリングした点を可視化
    plt.figure(figsize=(10, 2))
    plt.scatter(t_initial, x_initial, c=u_initial)
    plt.scatter(t_boundary, x_boundary, c=u_boundary)
    plt.colorbar()
    plt.scatter(t_region, x_region, c="gray", s=5, marker="x")
    plt.show()

    # 初期条件
    plt.figure(figsize=(3, 2))
    plt.scatter(x_initial, u_initial) # 初期条件
    plt.show()

    # 学習用のデータ
    X_bc = np.block([[t_initial, t_boundary], [x_initial, x_boundary]]).T #x軸のboundary condition
    Y_bc = np.block([[u_initial, u_boundary]]).T #y軸のboundary condition
    X_region = np.block([[t_region], [x_region]]).T 


    # modelの定義
    model = NN(2, 1)
    pinn_model = BurgersPINN(model)

    # モデルを学習
    history = pinn_model.fit(X_bc, Y_bc, X_region, max_epochs=MAX_EPOCHS_FOR_MODEL)

    # 履歴
    plt.figure(figsize=(10,2))
    plt.plot(history[:, 0], history[:, 1], label='loss_total')
    plt.plot(history[:, 0], history[:, 2], label='loss_u')
    plt.plot(history[:, 0], history[:, 3], label='loss_pi')
    plt.legend()


    ###### 予測 ######
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

    # 物理量を可視化
    plt.figure(figsize=(10, 2))
    plt.contourf(t_pred, x_pred, u_pred, 32)
    plt.colorbar()
    plt.show()

    # 基礎方程式の残差を可視化
    plt.figure(figsize=(10, 2))
    plt.contourf(t_pred, x_pred, f_pred, 32)
    plt.colorbar()
    plt.show()

    times = [0, 0.25, 0.5, 0.75]

    fig = plt.figure(figsize=(15, 2))

    for i, t in enumerate(times):
        t_pred = np.ones([100, 1]) * t
        X_in = np.column_stack([t_pred, x_pred])
        u_pred, f_pred = pinn_model.predict(X_in)
        
        fig.add_subplot(1, 4, i+1)    
        plt.plot(x_pred, u_pred, label='u_pred')
        plt.plot(x_pred, f_pred, label='f_pred')
        plt.ylim(-1.5, 1.5)
        plt.legend()
    plt.show()