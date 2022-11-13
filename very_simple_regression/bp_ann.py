import torch.nn as nn
import torch.nn.functional as F

class BPSimpleRegressor(nn.Module):
    def __init__(self):
        super(BPSimpleRegressor, self).__init__()
        self.linear_1 = nn.Linear(1, 3)
        self.linear_2 = nn.Linear(3, 1)
    
    def forward(self, input):
        out_1 = self.linear_1(input)
        out_1 = F.relu(out_1)
        out_2 = self.linear_2(out_1)
        out_2 = F.relu(out_2)
        return out_2