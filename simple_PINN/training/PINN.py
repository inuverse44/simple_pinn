import numpy as np
import torch
from simple_PINN.settings import config  # ← 修正ポイント

class PINN():
    def __init__(self, model):
        """
        PINNモデルの初期化
        @param model: PINNモデル
        """
        self.model = model
        self.velocity = config.get("VELOCITY")
        
    def net_u(self, x, t):
        u = self.model(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
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
        モデルを訓練
        """
        # configから取得（デフォルト）
        max_epochs = max_epochs or config.get("MAX_EPOCHS_FOR_FITTING")
        learning_rate = learning_rate or config.get("LEARNING_RATE")
        pi_weight = pi_weight or config.get("PI_WEIGHT")

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

            loss_u = torch.mean((u - u_pred)**2)

            if v_ic is not None:
                du_dt = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred),
                                            retain_graph=True, create_graph=True)[0]
                du_dt_ic = du_dt[:v_ic.shape[0]]
                loss_v = torch.mean((v_ic - du_dt_ic)**2)
            else:
                loss_v = 0.0

            loss_pi = torch.mean(f_pred**2)
            loss_total = loss_u + loss_v + pi_weight * loss_pi

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % config.get("EPOCH_SEPARATOR") == 0:
                print(f'epoch:{epoch}, \tloss:{loss_total.item()}, \tloss_u:{loss_u.item()}, \tloss_v:{loss_v if isinstance(loss_v, float) else loss_v.item()}, \tloss_pi:{loss_pi.item()}')
                history.append([epoch, loss_total.item(), loss_u.item(), loss_v.item(), loss_pi.item()])

        return np.array(history)

    def predict(self, X_in):
        t = torch.tensor(X_in[:, [0]], requires_grad=True).float()
        x = torch.tensor(X_in[:, [1]], requires_grad=True).float()

        self.model.eval()
        u = self.net_u(x, t).detach().numpy()
        f = self.net_f(x, t).detach().numpy()
        return u, f
