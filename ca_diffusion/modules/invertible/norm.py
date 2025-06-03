import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ca_diffusion.modules.utils import shape_val
from ca_diffusion.modules.invertible import InvertibleModule

class ActNorm(InvertibleModule):
    def __init__(self, channels, dim=1, eps=1e-6):
        super().__init__()

        assert dim>0 #we cannot choose dim=0, as it is alawys the batch dimension

        self.dim = dim
        self.channels = channels
        self.eps = eps
        self.mean = nn.Parameter(torch.zeros((channels,)))
        self.std = nn.Parameter(torch.ones((channels,)))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, x):
        with torch.no_grad():
            x = x.movedim(self.dim, 0).view(x.size(self.dim),-1)
            mean = torch.mean(x, dim=1)
            std = torch.std(x, dim=1)

            self.mean.copy_(mean)
            self.std.copy_(std)
            self.initialized.fill_(1)

    def forward(self, x, *args, **kwargs):
        if self.training and self.initialized.item() == 0:
            self.initialize(x)

        mean = shape_val(self.mean, x).movedim(0, self.dim)
        std = shape_val(self.std, x).movedim(0, self.dim)

        x = (x-mean)/(std+self.eps)

        #logdet
        shape = x.shape
        del shape[self.dim]
        factor = np.prod(shape[1:]) #scale by number of tokens (dim 0 is always batch size!!!)
        logdet = torch.sum(torch.log(torch.abs(self.scale)))*factor
        logdet = logdet*torch.ones((x.size(0),), device=self.scale.device)

        return x, logdet
        

    def backward(self, x, *args, **kwargs):
        mean = shape_val(self.mean, x).movedim(0, self.dim)
        std = shape_val(self.std, x).movedim(0, self.dim)

        x = x*(std+self.eps)+mean

        return x