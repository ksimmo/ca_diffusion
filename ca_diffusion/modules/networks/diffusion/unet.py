import numpy as np

import torch
import torch.nn as nn

from ca_diffusion.modules.utils import zero_init
from ca_diffusion.modules.conv.blocks import ResnetBlock2D, Attention2D


class Encoder(nn.Module):
    def __init__(self, channels_in, channels, channels_z, channel_mult=[2,4], num_blocks=2, attn_res=[], num_heads=8):
        super().__init__()

        self.channel_mult = [1] + channel_mult

        self.proj_in = nn.Conv2d(channels_in, channels, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()

        for i in range(len(self.channel_mult)-1):
            channels_a = np.power(2, self.channel_mult[i])*channels
            channels_b = np.power(2, self.channel_mult[i+1])*channels

            for j in range(num_blocks):
                self.downblocks.append(ResnetBlock2D(channels_a, channels_a, mode="default"))
                if i+1 in attn_res:
                    self.downblocks.append(Attention2D(channels_a, num_heads=num_heads))
            self.downblocks.append(ResnetBlock2D(channels_a, channels_b, mode="down"))

        for j in range(num_blocks):
            self.downblocks.append(ResnetBlock2D(channels_b, channels_b, mode="default"))
            if num_blocks in attn_res:
                self.downblocks.append(Attention2D(channels_b, num_heads=num_heads))
        self.downblocks.append(ResnetBlock2D(channels_b, channels_b, mode="default"))
        
        self.proj_out = nn.Sequential(nn.GroupNorm(32, channels_b), nn.SiLU(),
                                      nn.Conv2d(channels_b, channels_z, kernel_size=3, stride=1, padding=1, bias=True))


    def forward(self, x):
        x = self.proj_in(x)
        for block in self.downblocks:
            x = block(x)
        return self.proj_out(x)

        

class UNet(nn.Module):
    def __init__(self, channels_in, channels, channels_emb, channels_out=None, channel_mult=[2,4], num_blocks=2, attn_res=[], num_heads=8):
        super().__init__()

        self.channel_mult = [1] + channel_mult

        self.proj_in = nn.Conv2d(channels_in, channels, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        self.skips = nn.ModuleList()

        for i in range(len(self.channel_mult)-1):
            channels_a = np.power(2, self.channel_mult[i])*channels
            channels_b = np.power(2, self.channel_mult[i+1])*channels

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
            if num_blocks in attn_res:
                self.downblocks.append(Attention2D(channels_b, num_heads=num_heads))


        bottleneck = [ResnetBlock2D(channels_b, channels_b, mode="default", channels_emb=channels_emb)]
        if num_blocks in attn_res:
            bottleneck.append(Attention2D(channels_b, num_heads=num_heads))
        bottleneck += [ResnetBlock2D(channels_b, channels_b, mode="default", channels_emb=channels_emb)]
        self.bottleneck = nn.Sequential(*bottleneck)


        self.upblocks = nn.ModuleList()
        for j in range(num_blocks+1):
            self.upblocks.append(ResnetBlock2D(channels_b, channels_b, mode="default", channels_emb=channels_emb))
            if num_blocks in attn_res:
                self.upblocks.append(Attention2D(channels_b, num_heads=num_heads))
        for i in reversed(range(len(self.channel_mult)-1)):
            channels_b = np.power(2, self.channel_mult[i])*channels
            channels_a = np.power(2, self.channel_mult[i+1])*channels

            self.upblocks.append(ResnetBlock2D(channels_a, channels_b, mode="up", channels_emb=channels_emb))
            for j in range(num_blocks+1):
                self.upblocks.append(ResnetBlock2D(channels_b*2, channels_b, mode="default", channels_emb=channels_emb))
                if i+1 in attn_res:
                    self.upblocks.append(Attention2D(channels_b, num_heads=num_heads))

        self.proj_out = nn.Sequential(nn.GroupNorm(32, channels), nn.SiLU(), zero_init(nn.Conv3d(channels, channels_out, kernel_size=3, stride=1, padding=1, bias=True)))

    def forward(self, x, t):
        
        emb = None

        x = self.proj_in(x)

        skips = [x]
        for block in self.downblocks:
            x = block(x, emb)
            if not isinstance(block, Attention2D):
                skips.append(x)
        x = self.bottleneck(x)

        for block in self.upblocks:
            if isinstance(block, Attention2D):
                x = block(x, emb)
            elif hasattr(block, "mode"):
                if block.mode=="up":
                    x = block(x, emb)
            else:
                x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        
        x = self.proj_out(x)

        return x