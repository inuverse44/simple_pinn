import numpy as np
from simple_PINN.training.NN import NN
from simple_PINN.training.PINN import PINN
from simple_PINN.settings import config

def training():
    """
    Train the PINN model using data from the current configuration.

    This function performs:
    - Retrieval of initial, boundary, and interior points from config
    - Construction of input/output training data
    - Initialization and training of the PINN model
    - Returns the trained model and training history

    Returns:
        tuple:
            pinn_model (PINN): Trained Physics-Informed Neural Network.
            history (ndarray): Training loss history of shape [epochs, 5].
    """
    # Load training data from config
    t_initial = config.get("t_initial")
    x_initial = config.get("x_initial")
    u_initial = config.get("u_initial")
    v_initial = config.get("v_initial")

    t_boundary = config.get("t_boundary")
    x_boundary = config.get("x_boundary")
    u_boundary = config.get("u_boundary")

    t_region = config.get("t_region")
    x_region = config.get("x_region")

    max_epochs = config.get("MAX_EPOCHS_FOR_MODEL")

    # Combine initial and boundary conditions into training data
    X_bc = np.block([[t_initial, t_boundary], [x_initial, x_boundary]]).T  # shape: (N_bc, 2)
    Y_bc = np.block([[u_initial, u_boundary]]).T                          # shape: (N_bc, 1)
    V_ic = v_initial.reshape(-1, 1)                                       # shape: (N_initial, 1)
    X_region = np.block([[t_region], [x_region]]).T                       # shape: (N_region, 2)

    # Initialize model and PINN wrapper
    model = NN(2, 1)
    pinn_model = PINN(model)

    # Train the model
    history = pinn_model.fit(X_bc, Y_bc, X_region, v_ic=V_ic, max_epochs=max_epochs)

    return pinn_model, history
