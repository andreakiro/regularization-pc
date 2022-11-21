import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
sys.path.append("..")

from pc_blocks.pc_layer import PCLayer


class BPSimpleRegressor(nn.Module):
    """
    Simple ANN regressor, for backprop experiments.

    Parameters
    ----------
    dropout : Optional[float] (default is 0)
              dropout probability
    """
    def __init__(self, dropout: float = 0.0):
        super(BPSimpleRegressor, self).__init__()
        self.linear_1 = nn.Linear(1, 1024)
        self.linear_2 = nn.Linear(1024, 1024)
        self.linear_3 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=dropout)
    

    def forward(self, input) ->  torch.Tensor:
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

    Parameters
    ----------
    init    : str
              initialization technique PC hidden values; supported techniques:
                - 'zeros', hidden values initialized with 0s
                - 'normal', hidden values initialized with a normal distribution with μ=mean and σ=std
                - 'xavier_normal', hidden values initialize with values according to the method described in 
                  *Understanding the difficulty of training deep feedforward neural networks* - Glorot, X. & Bengio, Y. 
                  (2010), using a normal distribution. 
                - 'forward', hidden values initialized with the forward pass value
          
    dropout : Optional[float] (default is 0)
              dropout probability
    """
    def __init__(
        self, 
        init: str = 'forward', 
        dropout: float = 0.0
    ):
        super(PCSimpleRegressor, self).__init__()
        self.linear_1 = nn.Linear(1, 1024)
        self.pc_1 = PCLayer(size=1024, init=init)
        self.linear_2 = nn.Linear(1024, 1024)
        self.pc_2 = PCLayer(size=1024, init=init)
        self.linear_3 = nn.Linear(1024, 1)
        self.pc_3 = PCLayer(size=1, init=init)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, input) -> torch.Tensor:
        out_1 = self.linear_1(input)
        out_1 = self.dropout(out_1)
        out_1 = F.relu(out_1)
        out_1 = self.pc_1.forward(out_1)
        out_2 = self.linear_2(out_1)
        out_2 = self.dropout(out_2)
        out_2 = F.relu(out_2)
        out_2 = self.pc_2.forward(out_2)
        out_3 = self.linear_3(out_2)
        out_3 = self.pc_3.forward(out_3)
        return out_3