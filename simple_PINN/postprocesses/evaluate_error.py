from scipy.integrate import simpson
import numpy as np
import os

def evaluate_error(u_exact, u_pred, x_grid, t_grid, save_path=None):
    abs_diff = np.abs(u_exact - u_pred)
    sq_diff = (u_exact - u_pred)**2

    dx = x_grid[1, 0] - x_grid[0, 0]  # 修正: xは縦方向 (axis=0)
    dt = t_grid[0, 1] - t_grid[0, 0]  # 修正: tは横方向 (axis=1)

    # L1ノルム（2D積分）
    l1_error = simpson(simpson(abs_diff, dx=dt, axis=1), dx=dx, axis=0)

    # L2ノルム（二乗誤差の平方根）
    l2_squared = simpson(simpson(sq_diff, dx=dt, axis=1), dx=dx, axis=0)
    l2_error = np.sqrt(l2_squared)

    linf_error = np.max(abs_diff)

    # 表示
    print(f"L1 norm    : {l1_error:.5e}")
    print(f"L2 norm    : {l2_error:.5e}")
    print(f"Max error  : {linf_error:.5e}")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "a") as f:
            f.write("=== Evaluation Metrics ===\n")
            f.write(f"L1 norm    : {l1_error:.5e}\n")
            f.write(f"L2 norm    : {l2_error:.5e}\n")
            f.write(f"Max error  : {linf_error:.5e}\n")
            f.write("\n")

    return l1_error, l2_error, linf_error
