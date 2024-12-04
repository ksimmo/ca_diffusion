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