import numpy as np
import matplotlib.pyplot as plt

# 1次元のBurgers方程式
# 初期条件: x∈[-1,1], t=0のとき，u=-sin(x)
n_initial = 100
t_initial = np.zeros(n_initial) # t=[0, 0, ..., 0]
x_initial = (np.random.random(n_initial) - 0.5) * 2 #np.random.random()は[0, 1)の乱数を生成
u_initial = -1 * np.sin(np.pi * x_initial)

# 固定壁条件: x=-1 or x=1, t∈[0,1]のとき，u=0
n_boundary = 100
t_boundary = np.random.random(n_boundary)
x_boundary = np.random.choice([-1, 1], n_boundary)
u_boundary = np.zeros(n_boundary)

# 支配方程式の残差を評価する計算領域内の座標: x∈[-1,1], t∈[0,1]
n_region = 5000
t_region = np.random.random(n_region)
x_region = (np.random.random(n_region)- 0.5) * 2

# 最大エポック数
MAX_EPOCHS_FOR_MODEL = 1000
MAX_EPOCHS_FOR_FITTING = 300

# 学習率
LEARNING_RATE = 0.01

# pi項の重み
PI_WEIGHT = 5e-4

# モデルパラメータ
NU = 0.1