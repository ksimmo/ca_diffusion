import torch

from ca_diffusion.modules.invertible import InvertibleModule

class InvertibleLeakyReLU(InvertibleModule):
    def __init__(self, negative_slope=0.1):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, x, *args, **kwargs):
        x = torch.clamp(x,min=0)+torch.clamp(x,max=0)*self.negative_slope

        #logdet
        logdet = torch.ones((x.size(0),), device=x.device)

        return x, logdet
        

    def backward(self, x, *args, **kwargs):
        x = torch.clamp(x,min=0)+torch.clamp(x,max=0)/self.negative_slope
        return x