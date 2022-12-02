import torch

class PCLayer(torch.nn.Module):
    """
    Custom Predictive Coding Layer

    Parameters
    ----------
    size : int
           width of the previous output layer

    """
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size
        self.x = None
        self.ε = None 


    def forward(self, μ: torch.Tensor, init) -> torch.nn.Parameter:
        """
        Forward pass of the PC layer with optional initialization.

        In the forward pass of the PC layer, we want to detach the output coming from previous layers of the network 
        (or, more precisely, the previous PC layer guess to which is applied an affine transformationm and a non-linear 
        activation function, μ) from the value that is forwarded to the next layer (x). 
        Moreover, we want to compute and store the difference between the guessed value guessed from the previous layer 
        (μ), as we will differentiate afterwards w.r.t. these values in the energy descent (and weight update) steps.
        If init is set, the layer activation is also initialized.

        Parameters
        ----------
        μ : torch.Tensor 
            previous PC layer x value, to which is applied an affine transformation and an activation function

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
        Returns the current layer guess value.

        """
        if init is not None: self.init(init=init, μ=μ)
        self.ε = torch.sum(torch.square(self.x - μ), dim=1)
        return self.x


    def init(self, init, μ, mean=0.0, std=1.0, gain=1.0) -> None:
        """
        Initializes the activation of the layer neurons.

        Parameters
        ----------
        init : str (default is 'forward')
            initialization technique PC hidden values; supported techniques:
                - 'zeros', hidden values initialized with 0s
                - 'normal', hidden values initialized with a normal distribution with mean=μ and std=σ
                - 'xavier_normal', hidden values initialize with values according to the method described in 
                  *Understanding the difficulty of training deep feedforward neural networks* - Glorot, X. & Bengio, Y. 
                  (2010), using a normal distribution and gain=gain. 
                - 'forward', hidden values initialized with the forward pass value .

            
        μ : Optional[float] (default is 0.0)
            mean value used for 'normal' or 'forward' initialization. For 'forward', set μ to the forward pass value from the previous layer.

        σ : Optional[float] (default is 1.0)
            std value used for 'normal' initialization.

        gain: Optional[float] (default is 1.0)
            gain value used for 'xavier_normal' initialization.

        """
        if init == 'zeros':
            x = torch.zeros(μ.shape)
        elif init == 'normal':
            x = torch.empty(μ.shape)
            torch.nn.init.normal_(x, mean=mean, std=std)
        elif init == 'xavier_normal':
            x = torch.empty(μ.shape)
            torch.nn.init.xavier_normal_(x, gain=gain)
        elif init == 'forward':
            x = μ.clone().detach()
        else:
            raise ValueError(f"{init} is not a valid initialization technique!")
        self.x = torch.nn.Parameter(x)
