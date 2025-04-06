import numpy as np

# --- 初期条件 ---

N_INITIAL = 100
t_initial = np.zeros(N_INITIAL)  # t = 0
x_initial = np.linspace(-1, 1, N_INITIAL)
u_initial = np.sin(np.pi * x_initial) # 変位 u(x, 0) = sin(pi * x)
v_initial = np.zeros_like(x_initial) # 速度 ∂u/∂t(x, 0) = 0（静止）

N_BOUNDARY = 100
t_boundary = np.random.rand(N_BOUNDARY)
x_boundary = np.random.choice([-1, 1], N_BOUNDARY)
u_boundary = np.zeros(N_BOUNDARY)

N_REGION = 5000
t_region = np.random.rand(N_REGION)
x_region = np.random.uniform(-1, 1, N_REGION)


# --- HYPER PARAMETERS ---
MAX_EPOCHS_FOR_MODEL = 1000
MAX_EPOCHS_FOR_FITTING = 1000
LEARNING_RATE = 0.005
PI_WEIGHT = 1e-3
VELOCITY = 1

# --- その他の設定 ---
EPOCH_SEPARATOR = 10
