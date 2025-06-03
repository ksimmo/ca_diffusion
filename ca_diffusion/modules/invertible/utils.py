import torch

from ca_diffusion.modules.invertible import InvertibleModule

class Shuffle(InvertibleModule):
    def __init__(self, channels, dim=1):
        super().__init__()

        self.channels = channels
        self.dim = dim
        indices = torch.randperm(channels)
        self.register_buffer("forward_indices", indices)
        self.register_buffer("backward_indices", torch.argsort(indices))
    
    def forward(self, x, *args, **kwargs):
        x = x.movedim(self.dim, 0)
        x = x[self.forward_indices]
        x = x.movedim(0, self.dim)

        #logdet
        logdet = torch.ones((x.size(0),), device=x.device)

        return x, logdet
        

    def backward(self, x, *args, **kwargs):
        x = x.movedim(self.dim, 0)
        x = x[self.backward_indices]
        x = x.movedim(0, self.dim)
        return x