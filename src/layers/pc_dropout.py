import torch
from torch.autograd import Variable


class PCDropout(torch.nn.Module):
    r"""
    Custom Layer to implement Dropout used during PC Training

    Parameters
    ----------
    p : float
           Dropout probability

    """

    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self.mask = None

    def reset_mask(self) -> None:
        r"""
        Resets the current mask used to decide which signal is affected by Dropout

        """
        self.mask = None

    def forward(self, input: torch.Tensor, training: bool) -> torch.Tensor:
        r"""
        Applies Dropout on the input tensor and returns the resulting tensor.

        Parameters
        ----------
        input: torch.Tensor
                the input data on which to perform the dropout on.

        training : bool
            initialization technique PC hidden values; supported techniques:
                if True, Dropout will be applied
                if False, no Dropout will be applied

        Returns
        -------
        Returns the input Tensor applied with Dropout (during training)

        """
        if self.mask is None:
            self.mask = Variable(torch.bernoulli(
                input.data.new(input.data.size()).fill_(1 - self.p)))
        return (input * self.mask) / (1 - self.p) if training else input
