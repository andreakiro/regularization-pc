import torch

class PCLayer(torch.nn.Module):
    """
    Custom Predictive Coding Layer

    Parameters
    ----------
    size : int
           size of the input
    
    init : str
           initialization technique PC hidden values; supported techniques:
                - 'zeros', hidden values initialized with 0s
                - 'normal', hidden values initialized with a normal distribution with μ=mean and σ=std
                - 'xavier_normal', hidden values initialize with values according to the method described in 
                  *Understanding the difficulty of training deep feedforward neural networks* - Glorot, X. & Bengio, Y. 
                  (2010), using a normal distribution. 
                - 'forward', hidden values initialized with the forward pass value

    mean : Optional[float]
           mean value used for normal initialization; if None when using a normal initialization, is set to 0

    std : Optional[float]
           std value used for normal initialization; if None when using a normal initialization, is set to 1
    """
    def __init__(
        self, 
        size: int, 
        init: str,
        mean: float = None,
        std: float = None
    ) -> None:

        super().__init__()
        self.size = size
        self.ε = None
self.init = init
        x = torch.empty((1, self.size))

        if init == 'zeros':
            x = torch.zeros((1, self.size))

        elif init == 'normal':
            if mean is None: mean = 0
            if std is None: std = 1
            torch.nn.init.normal_(x, mean=mean, std=std)

        elif init == 'xavier_normal':
            torch.nn.init.xavier_normal_(x, gain=1.),

        elif init != 'forward':
            raise ValueError(f"{init} is not a valid initialization technique!")

        self.x = torch.nn.Parameter(x)



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
        if len(self.x.size()) == 0: 
            self.x = torch.mean(μ, dim=0, keepdim=True)  # forward pass initialization
        self.ε = (self.x - μ)**2
        return self.x
