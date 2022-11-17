import torch


class PCLayer(torch.nn.Module):
    """
    Custom Predictive Coding Layer

    Parameters
    ----------
    size : int
           size of the input
    
    init : str, optional (default is 'fwd')
           initialization technique, can be 'zeros', 'normal', 'fwd'
    """
    def __init__(self, size: int, init: str = 'fwd') -> None:

        super().__init__()

        self.size = size
        self.ε = None

        if init == 'zeros':
            x = torch.zeros((1, self.size))
        elif init == 'normal':
            x = torch.empty((1, self.size))
            torch.nn.init.xavier_normal_(x, gain=1., ),
        elif init != 'fwd':
            raise ValueError(f"{init} is not a valid initialization technique!")

        self.x = torch.nn.Parameter(x) if init != 'fwd' else None



    def forward(self, μ: torch.Tensor):
        """
        Forward pass of the PC layer.

        In the forward pass of the PC layer, we want to detach the output coming from previous layers of the network 
        (or, more precisely, the previous PC layer guess to which is applied an affine transformationm and a non-linear 
        activation function, μ) from the value that is forwarded to the next layer (x). 
        Moreover, we want to compute and store the difference between the guessed value guessed from the previous layer 
        (μ), as we will differentiate afterwards w.r.t. these values in the energy descent (and weight update?) steps.

        Parameters
        ----------
        μ : torch.Tensor 
            previous PC layer x value, to which is applied an affine transformation and an activation function

        Returns
        -------
        Returns the current layer guess value.

        """
        if self.x is None: 
            # forward pass initialization
            self.x = torch.mean(μ, dim=0, keepdim=True)  
        self.ε = (self.x - μ)**2
        return self.x
