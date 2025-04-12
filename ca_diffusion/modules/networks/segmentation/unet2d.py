import torch
import torch.nn as nn

from ca_diffusion.modules.utils import zero_init
from ca_diffusion.modules.conv.blocks2d import ResnetBlock2D, Attention2D

class UNet(nn.Module):
    def __init__(self, channels_in, channels, channels_out=None, channel_mult=[2,4], num_blocks=2, attn_res=[], num_heads=8, qkv_bias=True, qk_norm=True,
                 compile=False, use_checkpoint=False):
        super().__init__()

        channels_out = channels_in if channels_out is None else channels_out

        self.channel_mult = [1] + channel_mult

        self.proj_in = nn.Conv2d(channels_in, channels, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        skip_channels = [channels]

        for i in range(len(self.channel_mult)-1):
            channels_a = self.channel_mult[i]*channels
            channels_b = self.channel_mult[i+1]*channels

            for j in range(num_blocks):
                self.downblocks.append(ResnetBlock2D(channels_a, channels_a, mode="default", channels_emb=None, compile=compile, use_checkpoint=use_checkpoint))
                skip_channels.append(channels_a)
                #self.skips.append()
                if i+1 in attn_res:
                    self.downblocks.append(Attention2D(channels_a, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))
            self.downblocks.append(ResnetBlock2D(channels_a, channels_b, mode="down", channels_emb=None, compile=compile, use_checkpoint=use_checkpoint))
            skip_channels.append(channels_b)
            #self.skips.append()

        for j in range(num_blocks):
            self.downblocks.append(ResnetBlock2D(channels_b, channels_b, mode="default", channels_emb=None, compile=compile, use_checkpoint=use_checkpoint))
            #self.skips.append()
            skip_channels.append(channels_b)
            if -1 in attn_res:
                self.downblocks.append(Attention2D(channels_b, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))


        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(ResnetBlock2D(channels_b, channels_b, mode="default", channels_emb=None, compile=compile, use_checkpoint=use_checkpoint))
        if -1 in attn_res:
            self.bottleneck.append(Attention2D(channels_b, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))
        self.bottleneck.append(ResnetBlock2D(channels_b, channels_b, mode="default", channels_emb=None, compile=compile, use_checkpoint=use_checkpoint))


        self.upblocks = nn.ModuleList()
        for j in range(num_blocks+1):
            self.upblocks.append(ResnetBlock2D(channels_b+skip_channels.pop(), channels_b, mode="default", channels_emb=None, compile=compile, use_checkpoint=use_checkpoint))
            if -1 in attn_res:
                self.upblocks.append(Attention2D(channels_b, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))
        for i in reversed(range(len(self.channel_mult)-1)):
            channels_b = self.channel_mult[i]*channels
            channels_a = self.channel_mult[i+1]*channels

            self.upblocks.append(ResnetBlock2D(channels_a, channels_b, mode="up", channels_emb=None, compile=compile, use_checkpoint=use_checkpoint))
            for j in range(num_blocks+1):
                self.upblocks.append(ResnetBlock2D(channels_b+skip_channels.pop(), channels_b, mode="default", channels_emb=None, compile=compile, use_checkpoint=use_checkpoint))
                if i+1 in attn_res:
                    self.upblocks.append(Attention2D(channels_b, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))

        self.proj_out = nn.Sequential(nn.GroupNorm(32, channels), nn.SiLU(), zero_init(nn.Conv2d(channels, channels_out, kernel_size=3, stride=1, padding=1, bias=True)))

    def forward(self, x):
        x = self.proj_in(x)

        skips = [x]
        for block in self.downblocks:
            if not isinstance(block, Attention2D):
                x = block(x)
                skips.append(x)
            else:
                x = block(x)

        for block in self.bottleneck:
            if not isinstance(block, Attention2D):
                x = block(x)
            else:
                x = block(x)

        for block in self.upblocks:
            if isinstance(block, Attention2D):
                x = block(x)
            elif hasattr(block, "mode"):
                if block.mode=="up":
                    x = block(x)
                else:
                    x = torch.cat([x, skips.pop()], dim=1)
                    x = block(x) 
        
        x = self.proj_out(x)

        return x