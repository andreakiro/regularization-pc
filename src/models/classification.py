import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import PCLayer, PCSoftmaxLayer, PCDropout

INF = 1e+30  # infinity value, to reverse softmax


class BPSimpleClassifier(nn.Module):
    r"""
    Simple ANN classifier, for backprop experiments.
    
    Parameters
    ----------
    dropout : Optional[float] (default is 0.0)
              dropout probability

    """

    def __init__(self, dropout: float = 0) -> None:
        super(BPSimpleClassifier, self).__init__()
        # 922,240 free weight parameters
        self.linear_1 = nn.Linear(28*28, 512)
        self.linear_2 = nn.Linear(512, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.linear_4 = nn.Linear(256, 256)
        self.linear_5 = nn.Linear(256, 128)
        self.linear_6 = nn.Linear(128, 128)
        self.linear_7 = nn.Linear(128, 64)
        self.linear_8 = nn.Linear(64, 64)
        self.linear_9 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=dropout)
        self.linear_layers = [
            self.linear_1, 
            self.linear_2, 
            self.linear_3,
            self.linear_4,
            self.linear_5,
            self.linear_6,
            self.linear_7,
            self.linear_8,
            self.linear_9
        ]

    def forward(self, x) -> torch.Tensor:
        x = x.reshape(-1, 28*28)
        x = F.relu(self.dropout(self.linear_1(x)))
        x = F.relu(self.dropout(self.linear_2(x)))
        x = F.relu(self.dropout(self.linear_3(x)))
        x = F.relu(self.dropout(self.linear_4(x)))
        x = F.relu(self.dropout(self.linear_5(x)))
        x = F.relu(self.dropout(self.linear_6(x)))
        x = F.relu(self.dropout(self.linear_7(x)))
        x = F.relu(self.dropout(self.linear_8(x)))
        o = F.log_softmax(self.linear_9(x), dim=1)
        return o


class PCSimpleClassifier(nn.Module):
    r"""
    Simple ANN regressor, for predictive coding experiments. 

    Parameters
    ----------
    dropout : Optional[float] (default is 0)
              dropout probability

    """

    def __init__(self, dropout: float = 0.0) -> None:
        super(PCSimpleClassifier, self).__init__()
        # 922,240 free weight parameters
        self.linear_1 = nn.Linear(28*28, 512)
        self.linear_2 = nn.Linear(512, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.linear_4 = nn.Linear(256, 256)
        self.linear_5 = nn.Linear(256, 128)
        self.linear_6 = nn.Linear(128, 128)
        self.linear_7 = nn.Linear(128, 64)
        self.linear_8 = nn.Linear(64, 64)
        self.linear_9 = nn.Linear(64, 10)

        self.pc_layer1 = PCLayer(size=512)
        self.pc_layer2 = PCLayer(size=512)
        self.pc_layer3 = PCLayer(size=256)
        self.pc_layer4 = PCLayer(size=256)
        self.pc_layer5 = PCLayer(size=128)
        self.pc_layer6 = PCLayer(size=128)
        self.pc_layer7 = PCLayer(size=64)
        self.pc_layer8 = PCLayer(size=64)
        self.pc_layer9 = PCLayer(size=10)

        self.pc_softmax = PCSoftmaxLayer(size=10)

        self.pc_dropout1 = PCDropout(p=dropout)
        self.pc_dropout2 = PCDropout(p=dropout)
        self.pc_dropout3 = PCDropout(p=dropout)
        self.pc_dropout4 = PCDropout(p=dropout)
        self.pc_dropout5 = PCDropout(p=dropout)
        self.pc_dropout6 = PCDropout(p=dropout)
        self.pc_dropout7 = PCDropout(p=dropout)
        self.pc_dropout8 = PCDropout(p=dropout)

        self.dropout_layers = [
            self.pc_dropout1,
            self.pc_dropout2,
            self.pc_dropout3,
            self.pc_dropout4,
            self.pc_dropout5,
            self.pc_dropout6,
            self.pc_dropout7,
            self.pc_dropout8
        ]

        self.linear_layers = [
            self.linear_1,
            self.linear_2,
            self.linear_3,
            self.linear_4,
            self.linear_5,
            self.linear_6,
            self.linear_7,
            self.linear_8,
            self.linear_9,
        ]

        self.pc_layers = [
            self.pc_layer1,
            self.pc_layer2,
            self.pc_layer3, 
            self.pc_layer4,
            self.pc_layer5,
            self.pc_layer6,
            self.pc_layer7,
            self.pc_layer8,
            self.pc_layer9,
            self.pc_softmax
        ]

    def forward(self, input, init=None) -> torch.Tensor:
        r"""
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
        input = input.reshape(-1, 28*28)
        μ_1 = torch.relu(self.pc_dropout1(self.linear_1(input), self.training))
        x_1 = self.pc_layer1(μ_1, init) if self.training else μ_1
        μ_2 = torch.relu(self.pc_dropout2(self.linear_2(x_1), self.training))
        x_2 = self.pc_layer2(μ_2, init) if self.training else μ_2
        μ_3 = torch.relu(self.pc_dropout3(self.linear_3(x_2), self.training))
        x_3 = self.pc_layer3(μ_3, init) if self.training else μ_3
        μ_4 = torch.relu(self.pc_dropout4(self.linear_4(x_3), self.training))
        x_4 = self.pc_layer4(μ_4, init) if self.training else μ_4
        μ_5 = torch.relu(self.pc_dropout5(self.linear_5(x_4), self.training))
        x_5 = self.pc_layer5(μ_5, init) if self.training else μ_5
        μ_6 = torch.relu(self.pc_dropout6(self.linear_6(x_5), self.training))
        x_6 = self.pc_layer6(μ_6, init) if self.training else μ_6
        μ_7 = torch.relu(self.pc_dropout7(self.linear_7(x_6), self.training))
        x_7 = self.pc_layer7(μ_7, init) if self.training else μ_7
        μ_8 = torch.relu(self.pc_dropout8(self.linear_8(x_7), self.training))
        x_8 = self.pc_layer8(μ_8, init) if self.training else μ_8
        μ_9 = self.linear_9(x_8)
        x_9 = self.pc_layer9(μ_9, init) if self.training else μ_9
        return self.pc_softmax(x_9, init) if self.training else F.log_softmax(x_9, dim=1)

    def get_energy(self):
        r"""
        Returns the total energy of the network.

        """
        F = 0.0
        for l in self.pc_layers:
            F += l.ε
        return F

    def fix_output(self, output):
        r"""
        Sets the activation of the last pc_layer to the output.

        """
        output = F.one_hot(output, num_classes=10).type(torch.FloatTensor)
        output *= INF
        self.pc_layers[-1].x = torch.nn.Parameter(output)

    def reset_dropout_masks(self):
        r"""
        Resets the mask that was created to perform dropout on the signals.

        """
        for layer in self.dropout_layers:
            layer.reset_mask()
