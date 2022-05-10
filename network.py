import torch 
import torch.nn as nn

class FNN(nn.Module):
    """
    Fully connected neural networks
    """
    def __init__(self, layer_sizes, activation='tanh'):
        super(FNN, self).__init__()
        self.layer_sizes = layer_sizes 

        if activation == 'tanh':
            self.sigma = torch.tanh
        elif activation == 'sin':
            self.sigma = torch.sin 
        else:
            raise NotImplementedError(f'activation {activation} is not implemented till now')

        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            linear = nn.Linear(layer_sizes[i-1], layer_sizes[i])
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.linears.append(linear)
    
    def forward(self, x):
        for linear in self.linears[:-1]:
            x = linear(x)
            x = self.sigma(x)
        x = self.linears[-1](x)
        return x


class Network:
    """
    Build a physics informed neural network (PINN) model for the wave equation.
    """
    @classmethod
    def build(cls, num_inputs=2, layers=[32, 16, 16, 32], activation='tanh', num_outputs=1):
        """
        Build a PINN model for the wave equation with input shape (t, x) and output shape u(t, x).

        Args:
            num_inputs: number of input variables. Default is 2 for (t, x).
            layers: number of hidden layers.
            activation: activation function in hidden layers.
            num_outpus: number of output variables. Default is 1 for u(t, x).

        Returns:
            pytorch network model.
        """
        layer_sizes = [num_inputs] + layers + [1] 
        model = FNN(layer_sizes, activation=activation)
        return model 