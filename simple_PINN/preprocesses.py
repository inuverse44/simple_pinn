import numpy as np
import random
import torch
from settings import (
    N_INITIAL, 
    N_BOUNDARY, 
    N_REGION,
)

def torch_fix_seed(seed=42):
    """ 乱数シードを固定
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

# --- 初期条件 ---
t_initial = np.zeros(N_INITIAL)  # t = 0
x_initial = np.linspace(-1, 1, N_INITIAL)
u_initial = np.sin(np.pi * x_initial) # 変位 u(x, 0) = sin(pi * x)
v_initial = np.zeros_like(x_initial) # 速度 ∂u/∂t(x, 0) = 0（静止）

# --- 境界条件 ---
t_boundary = np.random.rand(N_BOUNDARY)
x_boundary = np.random.choice([-1, 1], N_BOUNDARY)
u_boundary = np.zeros(N_BOUNDARY)

# --- 計算領域内の点 ---
t_region = np.random.rand(N_REGION)
x_region = np.random.uniform(-1, 1, N_REGION)
