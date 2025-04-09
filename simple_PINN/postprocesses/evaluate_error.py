from scipy.integrate import simpson
import numpy as np
import os
from simple_PINN.settings import config  # ← 追加

def evaluate_error(u_exact, u_pred, x_grid, t_grid, save_path=None):
    abs_diff = np.abs(u_exact - u_pred)
    sq_diff = (u_exact - u_pred)**2

    dx = x_grid[1, 0] - x_grid[0, 0]  # x方向（行方向）
    dt = t_grid[0, 1] - t_grid[0, 0]  # t方向（列方向）

    # L1ノルム（2次元積分）
    l1_error = simpson(simpson(abs_diff, dx=dt, axis=1), dx=dx, axis=0)

    # L2ノルム（二乗誤差の平方根）
    l2_squared = simpson(simpson(sq_diff, dx=dt, axis=1), dx=dx, axis=0)
    l2_error = np.sqrt(l2_squared)

    # 無限ノルム（最大絶対値）
    linf_error = np.max(abs_diff)

    # 表示
    print(f"L1 norm    : {l1_error:.5e}")
    print(f"L2 norm    : {l2_error:.5e}")
    print(f"Max error  : {linf_error:.5e}")

    # 保存パスが未指定なら設定から取得
    if save_path is None:
        save_path = config.get_log_path()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "a") as f:
            f.write("=== Evaluation Metrics ===\n")
            f.write(f"L1 norm    : {l1_error:.5e}\n")
            f.write(f"L2 norm    : {l2_error:.5e}\n")
            f.write(f"Max error  : {linf_error:.5e}\n")
            f.write("\n")

    return l1_error, l2_error, linf_error
