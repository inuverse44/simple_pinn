import numpy as np
import matplotlib.pyplot as plt
from simple_PINN.settings import (
    t_initial, x_initial, u_initial, v_initial, 
    t_boundary, x_boundary, u_boundary,
    t_region, x_region, 
    TARGET_DIR
)

def sampling_points():
    """
    サンプリングした点を可視化
    """
    plt.figure(figsize=(10, 2))
    plt.scatter(t_initial, x_initial, c=u_initial)
    plt.scatter(t_boundary, x_boundary, c=u_boundary)
    plt.colorbar()
    plt.scatter(t_region, x_region, c="gray", s=5, marker="x")

    plt.title('Sampling Points')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')

    path = TARGET_DIR + "sampling_points.pdf"
    plt.savefig(path, bbox_inches='tight')  # 画像保存
    #plt.show()

def initial_conditions():
    """
    初期条件を可視化
    """
    plt.figure(figsize=(3, 2))
    plt.scatter(x_initial, u_initial) # 初期条件

    plt.title('Initial Condition')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')

    path = TARGET_DIR + "initial_conditions.pdf"
    plt.savefig(path, bbox_inches='tight')  # 画像保存
    #plt.show()

def loss(history):
    """
    lossの履歴を可視化
    """
    # 図のサイズを指定
    plt.figure(figsize=(10,2))

    # プロットするデータの指定
    plt.plot(history[:, 0], history[:, 1], label='loss_total')
    plt.plot(history[:, 0], history[:, 2], label='loss_u')
    plt.plot(history[:, 0], history[:, 3], label='loss_v')
    plt.plot(history[:, 0], history[:, 4], label='loss_pi')

    # 図の外観の設定
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss History')
    plt.legend()

    # 図を表示
    path = TARGET_DIR + "loss_history.pdf"
    plt.savefig(path, bbox_inches='tight')  # 画像保存
    #plt.show()

def prediction(t_pred, x_pred, u_pred):
    """
    予測した物理量を可視化
    """
    plt.figure(figsize=(10, 2))
    plt.contourf(t_pred, x_pred, u_pred, 32)
    plt.colorbar()

    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.title(r'predicted $u(t, x)$')

    path = TARGET_DIR + "prediction.pdf"
    plt.savefig(path, bbox_inches='tight')  # 画像保存
    #plt.show()

def residual(t_pred, x_pred, f_pred):
    """
    基礎方程式の残差を可視化
    """
    plt.figure(figsize=(10, 2))
    plt.contourf(t_pred, x_pred, f_pred, 32)
    plt.colorbar()

    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.title(r'residual $f(t, x)$')

    path = TARGET_DIR + "residual.pdf"
    plt.savefig(path, bbox_inches='tight')  # 画像保存
    #plt.show()

def time_evolution(pinn_model, x_pred):
    """
    物理量の時間変化を可視化
    """
    times = [0, 0.25, 0.5, 0.75, 1.0]
    fig = plt.figure(figsize=(15, 2))

    for i, t in enumerate(times):
        t_pred = np.ones([100, 1]) * t
        X_in = np.column_stack([t_pred, x_pred])
        u_pred, f_pred = pinn_model.predict(X_in)
        
        fig.add_subplot(1, len(times), i+1)    
        plt.plot(x_pred, u_pred, label='u_pred')
        plt.plot(x_pred, f_pred, label='f_pred')
        plt.ylim(-1.5, 1.5)
        plt.legend()

    path = TARGET_DIR + "time_evolution.pdf"
    plt.savefig(path, bbox_inches='tight')  # 画像保存

    #plt.show()



def plot_figures(pinn_model, history, t_pred, x_pred, u_pred, f_pred):
    """
    すべての図を描画
    """
    # サンプリングした点を可視化
    sampling_points()
    # 初期条件を可視化
    initial_conditions()
    # lossのh履歴を可視化
    loss(history)
    # 物理量を可視化
    prediction(t_pred, x_pred, u_pred)
    # 基礎方程式の残差を可視化
    residual(t_pred, x_pred, f_pred)
    # 物理量の時間変化を可視化
    time_evolution(pinn_model, x_pred)