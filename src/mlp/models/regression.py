import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

from src.layers import PCLayer


class BPSimpleRegressor(nn.Module):
    """
    Simple ANN regressor, for backprop experiments.

    Consists of an input and an output layer having width 1 and two hidden layers having width 1024.

    Parameters
    ----------
    dropout : Optional[float] (default is 0)
              dropout probability
    input_dim : Optional[int] (default is 1)
              the size of a 1d sample
    """
    def __init__(self, dropout: float = 0.0, input_dim: int = 1) -> None:
        super(BPSimpleRegressor, self).__init__()
        self.linear_1 = nn.Linear(input_dim, 1024)
        self.linear_2 = nn.Linear(1024, 1024)
        self.linear_3 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=dropout)
    

    def forward(self, x) ->  torch.Tensor:
        """
        Computes a forward pass through the network.

        Parameters
        ----------
        x: torch.Tensor
                the input data on which to compute the output.
        
        Returns
        -------
        Returns the output of the network as computed by forward propagation.
        
        """
        x = F.relu(self.dropout(self.linear_1(x)))
        x = F.relu(self.dropout(self.linear_2(x)))
        o = self.linear_3(x)
        return o


class PCSimpleRegressor(nn.Module):
    """
    Simple ANN regressor, for predictive coding experiments. 
    
    Consists of an input and an output layer having width 1 and two hidden layers having width 1024.
    Also has three pc layers that hold activations and prediction errors of the linear layers.

    Parameters
    ----------
    dropout : Optional[float] (default is 0)
              dropout probability
    input_dim : Optional[int] (default is 1)
              the size of a 1d sample
    """
    def __init__(self, dropout: float = 0.0, input_dim: int = 1) -> None:
        
        super(PCSimpleRegressor, self).__init__()
        self.linear_1 = nn.Linear(input_dim, 1024)
        self.linear_2 = nn.Linear(1024, 1024)
        self.linear_3 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=dropout)

        self.pc_layer1 = PCLayer(size=1024)
        self.pc_layer2 = PCLayer(size=1024)
        self.pc_layer3 = PCLayer(size=1)

        self.linear_layers = [self.linear_1, self.linear_2, self.linear_3]
        self.pc_layers = [self.pc_layer1, self.pc_layer2, self.pc_layer3]

        self.f = torch.relu
        self.f_prime = lambda x: torch.stack([torch.relu(torch.sign(torch.diag(x[i,:,0]))) for i in range(x.shape[0])])


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
        self.input = input
        μ_1 = self.f(self.dropout(self.linear_1(input)))
        x_1 = self.pc_layer1(μ_1, init) if self.training else μ_1
        μ_2 = self.f(self.dropout(self.linear_2(x_1)))
        x_2 = self.pc_layer2(μ_2, init) if self.training else μ_2
        μ_3 = self.linear_3(x_2)
        x_3 = self.pc_layer3(μ_3, init) if self.training else μ_3
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


    # gradients computation and local W, x updates

    def backward_x(self):
        self.grad_x = Parallel(n_jobs=len(self.pc_layers[:-1]))(delayed(self.grad_xi)(i) for i in range(len(self.pc_layers[:-1])))

    def backward_w(self):
        self.grad_w = Parallel(n_jobs=len(self.linear_layers))(delayed(self.grad_wi)(i) for i in range(len(self.pc_layers)))

    def step_x(self, η):
        new_x = Parallel(n_jobs=len(self.pc_layers[:-1]))(delayed(self.step_xi)(i, η) for i in range(len(self.pc_layers[:-1])))
        for l, x in zip(self.pc_layers[:-1], new_x):
            l.x = x

    def step_w(self, η):
        new_w = Parallel(n_jobs=len(self.linear_layers))(delayed(self.step_wi)(i, η) for i in range(len(self.linear_layers)))
        for l, w in zip(self.linear_layers, new_w):
            l.weight.data = w

    def grad_xi(self, i):
        with torch.no_grad():
            e_i, e_ii = self.pc_layers[i].ε.detach()[:, :, None], self.pc_layers[i+1].ε.detach()[:, :, None]
            w_ii = self.linear_layers[i+1].weight.detach()[None, :, :].expand(e_i.shape[0], -1, -1)
            x_i = self.pc_layers[i].x.detach()[:, :, None]
            grad = e_i - (torch.transpose(self.f_prime(w_ii @ x_i) @ w_ii, 1, 2) @ e_ii)
            return grad   

    def grad_wi(self, i):
        with torch.no_grad():
            e_i = self.pc_layers[i].ε.detach()[:, :, None]
            w_i = self.linear_layers[i].weight.detach()[None, :, :].expand(e_i.shape[0], -1, -1)
            x_i = self.pc_layers[i-1].x.detach()[:, :, None] if i > 0 else self.input.detach()[:, :, None]
            grad = self.f_prime(w_i @ x_i) @ e_i @ torch.transpose(x_i, 1, 2)
            return grad 

    def step_xi(self, i, η):
        with torch.no_grad():
            return torch.nn.Parameter(self.pc_layers[i].x - torch.matmul(self.grad_x[i], torch.Tensor([η])))

    def step_wi(self, i, η):
        with torch.no_grad():
            return torch.nn.Parameter(self.linear_layers[i].weight - torch.mul(torch.sum(self.grad_w[i], dim=0), η))


    # to use these gradients, add the following code to the trainer
    # # convergence step
    # for _ in range(self.iterations):

    #     self.model.forward(X_train)
    #     self.model.backward_x()
    #     self.model.step_x(η=0.2)

    # self.model.backward_w()
    # self.model.step_w(η=0.2)

    # and change the energy computation in the pc_layer
    # self.ε = torch.square(self.x - μ)
