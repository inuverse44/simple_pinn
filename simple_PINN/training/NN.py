import torch.nn as nn

class NN(nn.Module):
    """
    Fully connected neural network used in the PINN model.

    Architecture:
        Input layer → Multiple hidden layers with Tanh activation → Output layer

    Parameters:
        n_input (int): Number of input dimensions (e.g., 2 for (t, x)).
        n_output (int): Number of output dimensions (e.g., 1 for u).
        n_hiddens (list of int): Sizes of hidden layers. Default is [32, 64, 128, 64, 32].

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, n_output]
    """
    def __init__(self, n_input, n_output, n_hiddens=[32, 64, 128, 64, 32]):
        super(NN, self).__init__()

        # Activation function
        self.activation = nn.Tanh()

        # Input layer
        self.input_layer = nn.Linear(n_input, n_hiddens[0])

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(n_hiddens[i], n_hiddens[i + 1]) for i in range(len(n_hiddens) - 1)
        ])

        # Output layer
        self.output_layer = nn.Linear(n_hiddens[-1], n_output)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor of shape [batch_size, n_input]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_output]
        """
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x
