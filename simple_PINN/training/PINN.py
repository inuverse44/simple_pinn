import numpy as np
import torch
from simple_PINN.settings import config

class PINN:
    """
    Physics-Informed Neural Network (PINN) class for solving PDEs.

    This class includes methods to compute the network output (u),
    the residual of the governing PDE (f), train the model using boundary
    and initial conditions, and evaluate predictions.

    Attributes:
        model (nn.Module): Neural network used to approximate the solution.
        velocity (float): Wave propagation speed from config.
    """
    def __init__(self, model):
        """
        Initialize the PINN with a neural network model.

        Parameters:
            model (nn.Module): Neural network instance.
        """
        self.model = model
        self.velocity = config.get("VELOCITY")

    def net_u(self, x, t):
        """
        Predict u(t, x) using the neural network.

        Parameters:
            x (torch.Tensor): Input x-values of shape (N, 1).
            t (torch.Tensor): Input t-values of shape (N, 1).

        Returns:
            torch.Tensor: Predicted u-values of shape (N, 1).
        """
        return self.model(torch.cat([x, t], dim=1))

    def net_f(self, x, t):
        """
        Compute the residual of the wave equation:
            f = ∂²u/∂t² - c² ∂²u/∂x²

        Parameters:
            x (torch.Tensor): Input x-values (requires_grad=True).
            t (torch.Tensor): Input t-values (requires_grad=True).

        Returns:
            torch.Tensor: Residual tensor f(t, x).
        """
        u = self.net_u(x, t)
        du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        du_dtt = torch.autograd.grad(du_dt, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        du_dxx = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), retain_graph=True, create_graph=True)[0]

        f = du_dtt / (self.velocity ** 2) - du_dxx
        return f

    def fit(self, X_bc, Y_bc, X_region, v_ic=None,
            max_epochs=None, learning_rate=None, pi_weight=None):
        """
        Train the PINN model using boundary/initial data and residual minimization.

        Parameters:
            X_bc (ndarray): Boundary and initial input coordinates (t, x).
            Y_bc (ndarray): Boundary and initial true u values.
            X_region (ndarray): Collocation points in the domain (t, x).
            v_ic (ndarray, optional): Initial velocities ∂u/∂t at t=0.
            max_epochs (int, optional): Number of training epochs.
            learning_rate (float, optional): Learning rate for optimizer.
            pi_weight (float, optional): Weight for residual loss term.

        Returns:
            np.ndarray: Training loss history as array of [epoch, loss_total, loss_u, loss_v, loss_pi]
        """
        # Retrieve defaults from config if not specified
        max_epochs = max_epochs or config.get("MAX_EPOCHS_FOR_FITTING")
        learning_rate = learning_rate or config.get("LEARNING_RATE")
        pi_weight = pi_weight or config.get("PI_WEIGHT")

        # Prepare training tensors
        t = torch.tensor(X_bc[:, [0]], requires_grad=True).float()
        x = torch.tensor(X_bc[:, [1]], requires_grad=True).float()
        u = torch.tensor(Y_bc, requires_grad=True).float()
        t_region = torch.tensor(X_region[:, [0]], requires_grad=True).float()
        x_region = torch.tensor(X_region[:, [1]], requires_grad=True).float()
        if v_ic is not None:
            v_ic = torch.tensor(v_ic, requires_grad=True).float()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        history = []

        for epoch in range(max_epochs + 1):
            u_pred = self.net_u(x, t)
            f_pred = self.net_f(x_region, t_region)
            loss_u = torch.mean((u - u_pred) ** 2)

            if v_ic is not None:
                du_dt = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred),
                                            retain_graph=True, create_graph=True)[0]
                du_dt_ic = du_dt[:v_ic.shape[0]]
                loss_v = torch.mean((v_ic - du_dt_ic) ** 2)
            else:
                loss_v = 0.0

            loss_pi = torch.mean(f_pred ** 2)
            loss_total = loss_u + loss_v + pi_weight * loss_pi

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % config.get("EPOCH_SEPARATOR") == 0:
                print(f'epoch:{epoch}, \tloss:{loss_total.item():.4e}, \tloss_u:{loss_u.item():.4e}, '
                      f'\tloss_v:{loss_v if isinstance(loss_v, float) else loss_v.item():.4e}, \tloss_pi:{loss_pi.item():.4e}')
                history.append([epoch, loss_total.item(), loss_u.item(), loss_v.item(), loss_pi.item()])

        return np.array(history)

    def predict(self, X_in):
        """
        Predict u and residual f for input coordinates.

        Parameters:
            X_in (ndarray): Input coordinates (N, 2), where columns = [t, x].

        Returns:
            tuple: (u_pred, f_pred) as NumPy arrays of shape (N, 1)
        """
        t = torch.tensor(X_in[:, [0]], requires_grad=True).float()
        x = torch.tensor(X_in[:, [1]], requires_grad=True).float()

        self.model.eval()
        u = self.net_u(x, t).detach().numpy()
        f = self.net_f(x, t).detach().numpy()
        return u, f
