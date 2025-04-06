import torch.nn as nn

class NN(nn.Module):
    def __init__(self, n_input, n_output, n_hiddens=[32,64,128,64,32]):
        """ ニューラルネットワークの定義

        このクラスは、Burgers方程式のPINNモデルを定義します。

        :param n_input: 入力の次元数
        :type n_input: int
        :param n_output: 出力の次元数
        :type n_output: int
        :param n_hiddens: 隠れ層の数と次元数
        :type n_hiddens: list
        """

        # 親クラスの初期化
        # nn.Moduleを継承しているため、親クラスの初期化が必要
        super(NN, self).__init__() 
        
        # 隠れ層の数と次元数
        self.activation = nn.Tanh()
        
        self.input_layer = nn.Linear(n_input, n_hiddens[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(n_hiddens[i], n_hiddens[i+1]) for i in range(len(n_hiddens)-1)]
        )
        self.output_layer = nn.Linear(n_hiddens[-1], n_output)
        
    def forward(self, x):
        
        x = self.activation(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        x = self.output_layer(x)
        
        return x