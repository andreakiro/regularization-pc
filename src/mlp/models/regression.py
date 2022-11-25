import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
sys.path.append("..")

from src.pc_blocks.pc_layer import PCLayer


class BPSimpleRegressor(nn.Module):
    """
    Simple ANN regressor, for backprop experiments.

    Consists of three linear layers with input and output layer having width 1 and the hidden layer having width 1024.

    Parameters
    ----------
    dropout : Optional[float] (default is 0)
              dropout probability
    """
    def __init__(self, dropout: float = 0.0) -> None:
        super(BPSimpleRegressor, self).__init__()
        self.linear_1 = nn.Linear(1, 1024)
        self.linear_2 = nn.Linear(1024, 1024)
        self.linear_3 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=dropout)
    

    def forward(self, input) ->  torch.Tensor:
        """
        Computes a forward pass through the network.

        Parameters
        ----------
        input: torch.Tensor
                the input data on which to compute the output.
        
        Returns
        -------
        Returns the output of the network as computed by forward propagation.
        
        """
        out_1 = self.linear_1(input)
        out_1 = self.dropout(out_1)
        out_1 = F.relu(out_1)
        out_2 = self.linear_2(out_1)
        out_2 = self.dropout(out_2)
        out_2 = F.relu(out_2)
        out_3 = self.linear_3(out_2)
        return out_3


class PCSimpleRegressor(nn.Module):
    """
    Simple ANN regressor, for predictive coding experiments. 
    
    Consists of three linear layers with input and output layer having width 1 and the hidden layer having width 1024.
    Also has three pc layers that hold activations and prediction errors of the linear layers.

    Parameters
    ----------
    dropout : Optional[float] (default is 0)
              dropout probability
    """
    def __init__(self, dropout: float = 0.0) -> None:
        super(PCSimpleRegressor, self).__init__()
        self.linear_1 = nn.Linear(1, 1024)
        self.pc_1 = PCLayer(size=1024)
        self.linear_2 = nn.Linear(1024, 1024)
        self.pc_2 = PCLayer(size=1024)
        self.linear_3 = nn.Linear(1024, 1)
        self.pc_3 = PCLayer(size=1)
        
        self.linear_layers = [self.linear_1, self.linear_2, self.linear_3]
        self.pc_layers = [self.pc_1, self.pc_2, self.pc_3]

        self.dropout = nn.Dropout(p=dropout)


    def forward(self, input) -> torch.Tensor:
        """
        Computes a forward pass through the network.
        
        Note this works like standard forward propagation in BP networks.

        Parameters
        ----------
        input: torch.Tensor
                the input data on which to compute the output.
        
        Returns
        -------
        Returns the output of the network as computed by forward propagation.

        """
        μ_1 = self.linear_1(input)
        μ_1 = self.dropout(μ_1)
        μ_1 = torch.relu(μ_1)
        μ_2 = self.linear_2(μ_1)
        μ_2 = self.dropout(μ_2)
        μ_2 = torch.relu(μ_2)
        μ_3 = self.linear_3(μ_2)

        self.predicted_activations = [μ_1, μ_2, μ_3]

        return μ_3

    def pc_forward(self, input) -> torch.Tensor:
        μ_1 = self.linear_1(input)
        μ_1 = self.dropout(μ_1)
        μ_1 = torch.relu(μ_1)
        x_1 = self.pc_1(μ_1)
        μ_2 = self.linear_2(x_1)
        μ_2 = self.dropout(μ_2)
        μ_2 = torch.relu(μ_2)
        x_2 = self.pc_2(μ_2)
        μ_3 = self.linear_3(x_2)
        x_3 = self.pc_3(μ_3)

        return x_3
    

    def get_energy(self):
        """
        Returns the total energy of the network.

        """
        F = 0.0
        for l in self.pc_layers:
            F += l.ε
        return F
    
    def fix_output(self, output):
        """
        Sets the activation of the last pc_layer to the output.
        
        """
        self.pc_layers[-1].x = torch.nn.Parameter(output)
