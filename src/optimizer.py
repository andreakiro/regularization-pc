import torch


def set_optimizer(paramslist, optimizer, lr, wd, mom):
    r"""
    Selects the optimizer with its corresponding hyperparameters. 

    Parameters:
    ----------
        paramslist : torch.nn.ParameterList
                Model parameters (weights) that the optimizer will be updating.
        optmizer : str
                The type of the optimizer. Currently available: [adam, adagrad, rmsprop, momentum]
        lr : float
                The learning rate.
        wd : float
                Weight decay.
        mom : float
                momentum

    """
    optimizers = {
        'adam': torch.optim.Adam(
            params=paramslist,
            lr=lr,
            weight_decay=wd,
        ),

        'adagrad': torch.optim.Adagrad(
            params=paramslist,
            lr=lr,
            weight_decay=wd,
        ),

        'momentum': torch.optim.SGD(
            params=paramslist,
            lr=lr,
            momentum=mom,
            weight_decay=wd,
        ),

        'rmsprop': torch.optim.RMSprop(
            params=paramslist,
            lr=lr,
            momentum=mom,
            weight_decay=wd,
        ),
    }

    try:
        return optimizers[optimizer]
    except ValueError:
        return ValueError('Unknown optimizer.')
