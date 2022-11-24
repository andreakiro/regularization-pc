import torch.nn as nn
import torch.nn.functional as F
import torch

class BPClassifier(nn.Module):
    """
    Simple ANN classifier, for backprop experiments.

    Parameters
    ----------
    dropout : Optional[float] (default is 0.25)
              dropout probability
    """
    def __init__(self, dropout: float = 0.25):
        super(BPClassifier, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, 1)
        self.conv_2 = nn.Conv2d(32, 64, 3, 1)
        self.linear_1 = nn.Linear(9216, 128)
        self.linear_2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x) ->  torch.Tensor:
        # convolutionnal network part
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        # multi layer perceptron
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        output = F.log_softmax(x, dim=1)
        return output
    
class PCClassifier(nn.Module):

    def __init__(self):
        raise NotImplementedError