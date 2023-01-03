import torch
from torch.autograd import Variable


class PCDropout(torch.nn.Module):
    
    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self.mask = None


    def reset_mask(self):
        self.mask = None


    def forward(self, input: torch.Tensor, training: bool):
        if self.mask is None:
            self.mask = Variable(torch.bernoulli(input.data.new(input.data.size()).fill_(1 - self.p)))
        return (input * self.mask) / (1 - self.p) if training else input
