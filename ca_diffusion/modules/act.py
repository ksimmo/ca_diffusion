import torch
import torch.nn as nn
import torch.nn.functional as F

class GEGLU(nn.Module):
    def __init__(self, channels_in, channels_out, bias=True):
        super().__init__()
        self.proj = nn.Linear(channels_in, channels_out * 2, bias=bias)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
class SwiGLU(nn.Module):
    def __init__(self, channels_in, channels_out, bias=True):
        super().__init__()
        self.proj = nn.Linear(channels_in, channels_out * 2, bias=bias)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.silu(gate)

#Magnitude preserving SilU (EDM-2)
class MP_SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.silu(x)/0.596