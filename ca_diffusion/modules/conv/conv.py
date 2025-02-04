import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#normalize to unit magnitude
def normalize(x, eps=1e-4):
    dim = list(range(1, len(x.size())))
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    alpha = np.sqrt(n.numel(), x.numel())
    return x/torch.add(eps, n, alpha=alpha)

#Magnitude preserving 1D convolution (EDM-2)
class MP_Conv1d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0):
        super().__init__()

        assert isinstance(kernel_size, int)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn((channels_out, channels_in, self.kernel_size)))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight)/np.sqrt(fan_in)
        x = F.conv1d(x, w, stride=self.stride, padding=self.padding)
        return x

#Magnitude preserving 2D convolution (EDM-2)
class MP_Conv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=(3,3), stride=(1,1), padding=(0,0)):
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, list, set):
            assert len(kernel_size)==2
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn((channels_out, channels_in, self.kernel_size[0], self.kernel_size[1])))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight)/np.sqrt(fan_in)
        x = F.conv2d(x, w, stride=self.stride, padding=self.padding)
        return x
    
#Magnitude preserving 3D convolution (EDM-2)
class MP_Conv3d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,0,0)):
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        elif isinstance(kernel_size, list, set):
            assert len(kernel_size)==3
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn((channels_out, channels_in, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight)/np.sqrt(fan_in)
        x = F.conv3d(x, w, stride=self.stride, padding=self.padding)
        return x