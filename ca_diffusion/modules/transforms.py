import torch
import torch.nn as nn

from ca_diffusion.modules.utils import shape_val

class Transform(nn.Module):
    def __init__(self, method="identity", scale=1.0, offset=0.0, **kwargs):
        super().__init__()
        self.method = method

        if isinstance(scale, float):
            self.scale = scale
        elif isinstance(scale, (list, set)):
            self.register_buffer("scale", torch.FloatTensor(scale).unsqueeze(0))

        if isinstance(offset, float):
            self.offset = offset
        elif isinstance(offset, (list, set)):
            self.register_buffer("offset", torch.FloatTensor(scale).unsqueeze(0))

    def forward(self, x):
        if isinstance(self.scale, torch.Tensor):
            scale = shape_val(self.scale, x, -1) #we assume channel is at dim=1
        else:
            scale = self.scale
        if isinstance(self.offset, torch.Tensor):
            offset = shape_val(self.offset, x, -1)
        else:
            offset = self.offset

        x = x*scale+offset

        if self.method=="none" or self.method=="identity":
            pass
        elif self.method=="sqrt":
            x = torch.sign(x)*torch.sqrt(torch.abs(x))
        else:
            raise NotImplementedError("Transformation {} is not supported!".format(self.method))

        return x

    def backward(self, x):
        if isinstance(self.scale, torch.Tensor):
            scale = shape_val(self.scale, x, -1) #we assume channel is at dim=1
        else:
            scale = self.scale
        if isinstance(self.offset, torch.Tensor):
            offset = shape_val(self.offset, x, -1)
        else:
            offset = self.offset


        if self.method=="none" or self.method=="identity":
            pass
        elif self.method=="sqrt":
            x = torch.sign(x)*torch.pow(torch.abs(x), 2)
        else:
            raise NotImplementedError("Transformation {} is not supported!".format(self.method))

        x = (x-offset)/scale

        return x
