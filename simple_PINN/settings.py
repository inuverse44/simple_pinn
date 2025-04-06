import numpy as np

# --- 初期条件 ---
n_initial = 100
t_initial = np.zeros(n_initial)  # t = 0
x_initial = np.linspace(-1, 1, n_initial)
u_initial = np.sin(np.pi * x_initial) # 変位 u(x, 0) = sin(pi * x)
v_initial = np.zeros_like(x_initial) # 速度 ∂u/∂t(x, 0) = 0（静止）

# --- 境界条件 ---
n_boundary = 100
t_boundary = np.random.rand(n_boundary)
x_boundary = np.random.choice([-1, 1], n_boundary)
u_boundary = np.zeros(n_boundary)

# --- 計算領域内の点 ---
n_region = 5000
t_region = np.random.rand(n_region)
x_region = np.random.uniform(-1, 1, n_region)

# --- HYPER PARAMETERS ---
MAX_EPOCHS_FOR_MODEL = 100
MAX_EPOCHS_FOR_FITTING = 1000
LEARNING_RATE = 0.001
PI_WEIGHT = 5e-4
NU = 0.1  # 波動方程式では使わないけど削除不要

# --- その他の設定 ---
EPOCH_SEPARATOR = 10
