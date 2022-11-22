import torch.nn as nn
import torch.nn.functional as F

class BPSimpleRegressor(nn.Module):
    """
    Simple ANN regressor, for backprop experiments.
    """
    def __init__(self):
        super(BPSimpleRegressor, self).__init__()
        self.linear_1 = nn.Linear(1, 1024)
        self.linear_2 = nn.Linear(1024, 1024)
        self.linear_3 = nn.Linear(1024, 1)
    
    def forward(self, input):
        out_1 = self.linear_1(input)
        out_1 = F.relu(out_1)
        out_2 = self.linear_2(out_1)
        out_2 = F.relu(out_2)
        out_3 = self.linear_3(out_2)
        return out_3


class PCSimpleRegressor(nn.Module):
    """
    Simple ANN regressor, for predictive coding experiments.
    """
    def __init__(self):
        super(PCSimpleRegressor, self).__init__()
        raise NotImplementedError()
    
    def forward(self, input):
        raise NotImplementedError()