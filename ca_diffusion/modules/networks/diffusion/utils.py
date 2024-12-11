import numpy as np

import torch
import torch.nn as nn

from ca_diffusion.modules.utils import shape_val

class TimestepEmbedder(nn.Module):
    def __init__(self, channels_freq, channels, learnable=False):
        super().__init__()

        if learnable:
            self.freqs = nn.Parameter(torch.randn((1,channels_freq)))
            self.phases = nn.Parameter(torch.rand((1,channels_freq)))
        else:
            self.register_buffer("freqs", torch.randn((1,channels_freq)))
            self.register_buffer("phases", torch.rand((1,channels_freq)))

        self.embedder = nn.Sequential(nn.Linear(channels_freq, channels), nn.SiLU(), nn.Linear(channels,channels))

    def forward(self, t):
        freqs = shape_val(self.freqs.to(torch.float32), t, pos=0)
        phases = shape_val(self.phases.to(torch.float32), t, pos=0)
        emb = torch.cos((freqs*t.to(torch.float32).unsqueeze(-1)+phases)*2.0*np.pi)
        return self.embedder(emb.to(t.dtype))