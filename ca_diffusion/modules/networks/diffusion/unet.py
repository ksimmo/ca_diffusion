import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ca_diffusion.modules.utils import zero_init, shape_val
from ca_diffusion.modules.conv.blocks import ResnetBlock2D, Attention2D


class Encoder(nn.Module):
    def __init__(self, channels_in, channels, channels_z, channel_mult=[2,4,], num_blocks=2, attn_res=[], num_heads=8):
        super().__init__()

        self.channel_mult = [1] + channel_mult

        self.proj_in = nn.Conv2d(channels_in, channels, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()

        for i in range(len(self.channel_mult)-1):
            channels_a = self.channel_mult[i]*channels
            channels_b = self.channel_mult[i+1]*channels

            for j in range(num_blocks):
                self.downblocks.append(ResnetBlock2D(channels_a, channels_a, mode="default"))
                if i+1 in attn_res:
                    self.downblocks.append(Attention2D(channels_a, num_heads=num_heads))
            self.downblocks.append(ResnetBlock2D(channels_a, channels_b, mode="down"))

        for j in range(num_blocks):
            self.downblocks.append(ResnetBlock2D(channels_b, channels_b, mode="default"))
            if -1 in attn_res:
                self.downblocks.append(Attention2D(channels_b, num_heads=num_heads))
        self.downblocks.append(ResnetBlock2D(channels_b, channels_b, mode="default"))
        
        self.proj_out = nn.Sequential(nn.GroupNorm(32, channels_b), nn.SiLU(),
                                      nn.Conv2d(channels_b, channels_z, kernel_size=3, stride=1, padding=1, bias=True))


    def forward(self, x):
        x = self.proj_in(x)
        for block in self.downblocks:
            x = block(x)
        return self.proj_out(x)
    

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
        freqs = shape_val(self.freqs, t, pos=0)
        phases = shape_val(self.phases, t, pos=0)
        emb = torch.cos((freqs*t.unsqueeze(-1)+phases)*2.0*np.pi)
        return self.embedder(emb)

        

class UNet(nn.Module):
    def __init__(self, channels_in, channels, channels_emb, channels_out=None, channel_mult=[2,4], num_blocks=2, attn_res=[], num_heads=8,
                 channels_freq=256):
        super().__init__()

        channels_out = channels_in if channels_out is None else channels_out

        self.channel_mult = [1] + channel_mult

        self.proj_in = nn.Conv2d(channels_in, channels, kernel_size=3, stride=1, padding=1)

        self.timestep_embedder = TimestepEmbedder(channels_freq, channels_emb, learnable=True)

        self.downblocks = nn.ModuleList()
        self.skips = nn.ModuleList()

        for i in range(len(self.channel_mult)-1):
            channels_a = self.channel_mult[i]*channels
            channels_b = self.channel_mult[i+1]*channels

            for j in range(num_blocks):
                self.downblocks.append(ResnetBlock2D(channels_a, channels_a, mode="default", channels_emb=channels_emb))
                #self.skips.append()
                if i+1 in attn_res:
                    self.downblocks.append(Attention2D(channels_a, num_heads=num_heads))
            self.downblocks.append(ResnetBlock2D(channels_a, channels_b, mode="down", channels_emb=channels_emb))
            #self.skips.append()

        for j in range(num_blocks):
            self.downblocks.append(ResnetBlock2D(channels_b, channels_b, mode="default", channels_emb=channels_emb))
            #self.skips.append()
            if -1 in attn_res:
                self.downblocks.append(Attention2D(channels_b, num_heads=num_heads))


        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(ResnetBlock2D(channels_b, channels_b, mode="default", channels_emb=channels_emb))
        if -1 in attn_res:
            self.bottleneck.append(Attention2D(channels_b, num_heads=num_heads))
        self.bottleneck.append(ResnetBlock2D(channels_b, channels_b, mode="default", channels_emb=channels_emb))


        self.upblocks = nn.ModuleList()
        for j in range(num_blocks+1):
            self.upblocks.append(ResnetBlock2D(channels_b, channels_b, mode="default", channels_emb=channels_emb))
            if -1 in attn_res:
                self.upblocks.append(Attention2D(channels_b, num_heads=num_heads))
        for i in reversed(range(len(self.channel_mult)-1)):
            channels_b = self.channel_mult[i]*channels
            channels_a = self.channel_mult[i+1]*channels

            self.upblocks.append(ResnetBlock2D(channels_a, channels_b, mode="up", channels_emb=channels_emb))
            for j in range(num_blocks+1):
                self.upblocks.append(ResnetBlock2D(channels_b*2, channels_b, mode="default", channels_emb=channels_emb))
                if i+1 in attn_res:
                    self.upblocks.append(Attention2D(channels_b, num_heads=num_heads))

        self.proj_out = nn.Sequential(nn.GroupNorm(32, channels), nn.SiLU(), zero_init(nn.Conv2d(channels, channels_out, kernel_size=3, stride=1, padding=1, bias=True)))

    def forward(self, x, t):
        
        emb = self.timestep_embedder(t)
        emb = F.silu(emb)

        x = self.proj_in(x)

        skips = [x]
        for block in self.downblocks:
            if not isinstance(block, Attention2D):
                x = block(x, emb)
                skips.append(x)
            else:
                x = block(x)

        for block in self.bottleneck:
            if not isinstance(block, Attention2D):
                x = block(x, emb)
            else:
                x = block(x)

        for block in self.upblocks:
            if isinstance(block, Attention2D):
                x = block(x)
            elif hasattr(block, "mode"):
                if block.mode=="up":
                    x = block(x, emb)
            else:
                x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        
        x = self.proj_out(x)

        return x