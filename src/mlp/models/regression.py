import torch.nn as nn
import torch.nn.functional as F
import torch

from src.layers import PCLayer


class BPSimpleRegressor(nn.Module):
    """
    Simple ANN regressor, for backprop experiments.

    Consists of an input and an output layer having width 1 and two hidden layers having width 1024.

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
    
    Consists of an input and an output layer having width 1 and two hidden layers having width 1024.
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


    def forward(self, input, init=None) -> torch.Tensor:
        """
        Computes a forward pass through the network. 
        
        If the model is in training mode, the forward pass is a predictive coding forward pass. Note this only does local computations.
        If the model is in eval mode, the forward pass is a regular forward pass (cf. backpropagation). 
        If init is set, it initializes the activations of the pc layers. 
        
        Parameters
        ----------
        input: torch.Tensor
                the input data on which to compute the output.
        
        init : str (default is 'forward')
            initialization technique PC hidden values; supported techniques:
                - 'zeros', hidden values initialized with 0s
                - 'normal', hidden values initialized with a normal distribution with mean=μ and std=σ
                - 'xavier_normal', hidden values initialize with values according to the method described in 
                  *Understanding the difficulty of training deep feedforward neural networks* - Glorot, X. & Bengio, Y. 
                  (2010), using a normal distribution and gain=gain. 
                - 'forward', hidden values initialized with the forward pass value .
        
        Returns
        -------
        Returns the activation of the output neuron, i.e. the last pc layer.
        
        """
        μ_1 = self.linear_1(input)
        μ_1 = self.dropout(μ_1)
        μ_1 = torch.relu(μ_1)
        x_1 = self.pc_1(μ_1, init) if self.training else μ_1
        μ_2 = self.linear_2(x_1)
        μ_2 = self.dropout(μ_2)
        μ_2 = torch.relu(μ_2)
        x_2 = self.pc_2(μ_2, init) if self.training else μ_2
        μ_3 = self.linear_3(x_2)
        x_3 = self.pc_3(μ_3, init) if self.training else μ_3

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
