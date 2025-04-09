import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from simple_PINN.settings.config import (
    t_initial, x_initial, u_initial, v_initial, 
    t_boundary, x_boundary, u_boundary,
    t_region, x_region, 
    get_target_dir
)
TARGET_DIR = get_target_dir()

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

    # 共通の凡例を下部に追加
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))

    fig.tight_layout()
    path = TARGET_DIR + "time_evolution.pdf"
    plt.savefig(path, bbox_inches='tight')

import matplotlib.pyplot as plt
import numpy as np
from simple_PINN.settings.config import TARGET_DIR

def difference(pinn_model, x_pred):
    """
    物理量の時間変化における u_exact - u_pred の差分を可視化
    """
    times = [0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(times), figsize=(15, 2), sharey=True)

    for i, t in enumerate(times):
        t_pred = np.ones([100, 1]) * t
        X_in = np.column_stack([t_pred, x_pred])
        u_pred, _ = pinn_model.predict(X_in)
        u_exact = np.sin(np.pi * x_pred.flatten()) * np.cos(np.pi * t)
        diff_u = u_pred.flatten() - u_exact
        diff_u2 = - diff_u
        ax = axes[i]
        ax.plot(x_pred, diff_u, label=r'$u_{\rm exact} - u_{\rm pred}$',
                linewidth=1, linestyle='solid', color='black')
        ax.plot(x_pred, diff_u2, linewidth=1, linestyle='dashed', color='black')
        ax.set_ylim(0, 1.5)
        ax.set_title(f"t = {t:.2f}")

        if i == 0:
            ax.set_ylabel("difference")

    # 共通の凡例を下部に表示
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=1, bbox_to_anchor=(0.5, -0.15))

    fig.tight_layout()
    path = TARGET_DIR + "difference.pdf"
    plt.savefig(path, bbox_inches='tight')


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
    # 物理量の時間変化とexact solutionの差を可視化
    difference(pinn_model, x_pred)
    