import numpy as np
from simple_PINN.training.NN import NN
from simple_PINN.training.PINN import PINN
from simple_PINN.settings.config import ( 
    t_initial, x_initial, u_initial, v_initial, 
    t_boundary, x_boundary, u_boundary,
    t_region, x_region, 
    MAX_EPOCHS_FOR_MODEL, 
)

def training():
    # data for training
    # 注意：初期条件（t=0）と境界条件（t∈[0,1]）を合成してX_bcにしているため、tやuも「初期条件+境界条件=200個」になっている
    X_bc = np.block([[t_initial, t_boundary], [x_initial, x_boundary]]).T #x軸のboundary condition
    Y_bc = np.block([[u_initial, u_boundary]]).T #y軸のboundary condition
    V_ic = v_initial.reshape(-1, 1) # 初期条件の速度(t=0)
    X_region = np.block([[t_region], [x_region]]).T 

    model = NN(2, 1)
    pinn_model = PINN(model)

    # the model is trained with the boundary condition
    history = pinn_model.fit(X_bc, Y_bc, X_region, v_ic=V_ic, max_epochs=MAX_EPOCHS_FOR_MODEL)
    return pinn_model, history
