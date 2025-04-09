
import numpy as np
import torch
from simple_PINN.settings.config import (
    MAX_EPOCHS_FOR_FITTING,
    LEARNING_RATE, 
    PI_WEIGHT,
    VELOCITY, 
    EPOCH_SEPARATOR
)

class PINN():
    def __init__(self, model):
        """
        PINNモデルの初期化
        @param model: PINNモデル
        @param NU: 粘性係数
        """
        self.model = model
        self.velocity = VELOCITY
        
        
    def net_u(self, x, t):
        """ 
        物理量を出力
        """
        u = self.model(torch.cat([x, t], dim=1))
        return u


    def net_f(self, x, t):
        """ 
        支配方程式との残差を出力
        """    
        # モデルが予測する物理量
        u = self.net_u(x, t)
        
        # 微分係数を自動微分で計算
        du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        du_dtt = torch.autograd.grad(du_dt, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        du_dxx = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), retain_graph=True, create_graph=True)[0]
        
        # 支配方程式に代入(f=0だと方程式と完全一致)
        coefficient = self.velocity**2
        f = du_dtt / coefficient - du_dxx
        
        return f
    
    
    def fit(self, X_bc, Y_bc, X_region, v_ic=None, max_epochs=MAX_EPOCHS_FOR_FITTING, learning_rate=LEARNING_RATE, pi_weight=PI_WEIGHT):
        """ 
        学習データでモデルを訓練
        """
        # 入力データをスライス
        t = X_bc[:, [0]]
        x = X_bc[:, [1]]
        u = Y_bc 
        t_region = X_region[:, [0]] 
        x_region = X_region[:, [1]]
        
        # 入力をtorch.tensorに変換
        t = torch.tensor(t, requires_grad=True).float()
        x = torch.tensor(x, requires_grad=True).float()
        u = torch.tensor(u, requires_grad=True).float()
        x_region = torch.tensor(x_region, requires_grad=True).float()
        t_region = torch.tensor(t_region, requires_grad=True).float()
        if v_ic is not None:
            v_ic = torch.tensor(v_ic, requires_grad=True).float()
        
        # 最適化ロジック（Adamを使用）
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # モデルを学習モードに変更
        self.model.train()
        
        # 学習
        history = []
        for epoch in range(max_epochs + 1):
            u_pred = self.net_u(x, t)
            f_pred = self.net_f(x_region, t_region)

            # uの損失 (平均二乗誤差)
            loss_u = torch.mean((u - u_pred)**2)

            # ∂u/∂t(x, 0) の損失（t = 0 の点だけを選択）
            if v_ic is not None:
                du_dt = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred),
                                            retain_graph=True, create_graph=True)[0]
                n_ic = v_ic.shape[0] # 0の時刻の点の数
                du_dt_ic = du_dt[:n_ic] # 0の時刻の点だけを選択
                loss_v = torch.mean((v_ic - du_dt_ic)**2)
            else:
                loss_v = 0.0

            # 方程式の残差の損失
            loss_pi = torch.mean(f_pred**2)

            # 総損失
            loss_total = loss_u + loss_v + loss_pi * pi_weight

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % EPOCH_SEPARATOR == 0:
                print(f'epoch:{epoch}, \tloss:{loss_total.item()}, \tloss_u:{loss_u.item()}, \tloss_v:{loss_v if isinstance(loss_v, float) else loss_v.item()}, \tloss_pi:{loss_pi.item()}')
                history.append([epoch, loss_total.item(), loss_u.item(), loss_v.item(), loss_pi.item()])
            
        return np.array(history)
                
    def predict(self, X_in):
        """ 
        モデルの予測
        """
        t = X_in[:, [0]]
        x = X_in[:, [1]]
        
        # 入力をtorch.tensorに変換
        t = torch.tensor(t, requires_grad=True).float()
        x = torch.tensor(x, requires_grad=True).float()
        
        self.model.eval()  # 評価モードに変更

        u = self.net_u(x, t).detach().numpy()
        f = self.net_f(x, t).detach().numpy()
        
        return u, f