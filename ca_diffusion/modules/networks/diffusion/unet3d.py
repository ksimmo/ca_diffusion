import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ca_diffusion.modules.utils import zero_init, shape_val
from ca_diffusion.modules.conv.blocks3d import Conv3D, ResnetBlock3D, Attention3D
from ca_diffusion.modules.networks.diffusion.utils import TimestepEmbedder


class Encoder(nn.Module):
    def __init__(self, channels_in, channels, channels_z, channel_mult=[2,4], num_blocks=2, attn_res=[], num_heads=8, qkv_bias=True, qk_norm=True, doublez=False,
                 compile=False, use_checkpoint=False, causal=False, learnable_pool=False, mode_res=None):
        super().__init__()

        self.channel_mult = [1] + channel_mult

        self.proj_in = Conv3D(channels_in, channels, kernel_size=3, stride=1, padding=1, causal=causal)

        self.downblocks = nn.ModuleList()

        for i in range(len(self.channel_mult)-1):
            channels_a = self.channel_mult[i]*channels
            channels_b = self.channel_mult[i+1]*channels

            if mode_res is None:
                encoder_mode = ""
            else:
                encoder_mode = "_"+mode_res[i] if mode_res[i] != "" else ""

            for j in range(num_blocks):
                self.downblocks.append(ResnetBlock3D(channels_a, channels_a, mode="default", compile=compile, use_checkpoint=use_checkpoint,
                                                     causal=causal, learnable_pool=learnable_pool))
                if i+1 in attn_res:
                    self.downblocks.append(Attention3D(channels_a, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))
            self.downblocks.append(ResnetBlock3D(channels_a, channels_b, mode="down"+encoder_mode, compile=compile, use_checkpoint=use_checkpoint,
                                                 causal=causal, learnable_pool=learnable_pool))


        for j in range(num_blocks):
            self.downblocks.append(ResnetBlock3D(channels_b, channels_b, mode="default", compile=compile, use_checkpoint=use_checkpoint,
                                                 causal=causal, learnable_pool=learnable_pool))
            if -1 in attn_res:
                self.downblocks.append(Attention3D(channels_b, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))
        
        self.proj_out = nn.Sequential(nn.GroupNorm(32, channels_b), nn.SiLU(),
                                      Conv3D(channels_b, channels_z*2 if doublez else channels_z, kernel_size=3, stride=1, padding=1, bias=True, causal=causal))


    def forward(self, x):
        x = self.proj_in(x)
        for block in self.downblocks:
            x = block(x)
        return self.proj_out(x)
    
    
class Decoder(nn.Module):
    def __init__(self, channels_out, channels, channels_z, channel_mult=[2,4], num_blocks=2, attn_res=[], num_heads=8, qkv_bias=True, qk_norm=True,
                 compile=False, use_checkpoint=False, causal=False, learnable_pool=False, mode_res=None):
        super().__init__()

        self.channel_mult = [1] + channel_mult

        channels_a = self.channel_mult[-1]*channels
        self.proj_in = Conv3D(channels_z, channels_a, kernel_size=3, stride=1, padding=1, causal=causal)

        self.upblocks = nn.ModuleList()
        for j in range(num_blocks):
            self.upblocks.append(ResnetBlock3D(channels_b, channels_b, mode="default", compile=compile, use_checkpoint=use_checkpoint,
                                               causal=causal, learnable_pool=learnable_pool))
            if -1 in attn_res:
                self.upblocks.append(Attention3D(channels_b, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))
        for i in reversed(range(len(self.channel_mult)-1)):
            channels_b = self.channel_mult[i]*channels
            channels_a = self.channel_mult[i+1]*channels

            if mode_res is None:
                decoder_mode = ""
            else:
                decoder_mode = "_"+mode_res[i] if mode_res[i] != "" else ""

            self.upblocks.append(ResnetBlock3D(channels_a, channels_b, mode="up"+decoder_mode, compile=compile, use_checkpoint=use_checkpoint,
                                               causal=causal, learnable_pool=learnable_pool))
            for j in range(num_blocks):
                self.upblocks.append(ResnetBlock3D(channels_b, channels_b, mode="default", compile=compile, use_checkpoint=use_checkpoint,
                                                   causal=causal, learnable_pool=learnable_pool))
                if i+1 in attn_res:
                    self.upblocks.append(Attention3D(channels_b, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))

        self.proj_out = nn.Sequential(nn.GroupNorm(32, channels), nn.SiLU(), zero_init(Conv3D(channels, channels_out, kernel_size=3, stride=1, padding=1, bias=True, causal=causal)))


    def forward(self, x):
        x = self.proj_in(x)
        for block in self.upblocks:
            x = block(x)
        return self.proj_out(x)

        

class UNet(nn.Module):
    def __init__(self, channels_in, channels, channels_emb, channels_out=None, channel_mult=[2,4], num_blocks=2, attn_res=[], num_heads=8, qkv_bias=True, qk_norm=True,
                 channels_freq=256, compile=False, use_checkpoint=False, channels_injection=None, injection_res=None, causal=False, learnable_pool=False,
                 mode_res=None):
        super().__init__()

        channels_out = channels_in if channels_out is None else channels_out

        self.channel_mult = [1] + channel_mult

        self.proj_in = Conv3D(channels_in, channels, kernel_size=3, stride=1, padding=1, causal=causal)

        self.timestep_embedder = TimestepEmbedder(channels_freq, channels_emb, learnable=True)

        self.downblocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        skip_channels = [channels]

        self.use_injection = channels_injection is not None
        self.injection_proj = None
        self.injection_res = injection_res

        for i in range(len(self.channel_mult)-1):
            channels_a = self.channel_mult[i]*channels
            channels_b = self.channel_mult[i+1]*channels

            if mode_res is None:
                encoder_mode = ""
            else:
                encoder_mode = "_"+mode_res[i] if mode_res[i] != "" else ""

            if self.use_injection and i+1==injection_res:
                self.injection_proj = Conv3D(channels_injection, channels_a, kernel_size=3, stride=1, padding=1, bias=True, causal=causal)

            for j in range(num_blocks):
                if self.use_injection and i+1==injection_res and j==0:
                    self.downblocks.append(ResnetBlock3D(channels_a*2, channels_a, mode="default", channels_emb=channels_emb, compile=compile, use_checkpoint=use_checkpoint,
                                                         causal=causal, learnable_pool=learnable_pool))
                else:
                    self.downblocks.append(ResnetBlock3D(channels_a, channels_a, mode="default", channels_emb=channels_emb, compile=compile, use_checkpoint=use_checkpoint,
                                                         causal=causal, learnable_pool=learnable_pool))
                skip_channels.append(channels_a)
                #self.skips.append()
                if i+1 in attn_res:
                    self.downblocks.append(Attention3D(channels_a, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))
            self.downblocks.append(ResnetBlock3D(channels_a, channels_b, mode="down"+encoder_mode, channels_emb=channels_emb, compile=compile, use_checkpoint=use_checkpoint,
                                                 causal=causal, learnable_pool=learnable_pool))
            skip_channels.append(channels_b)
            #self.skips.append()

        for j in range(num_blocks):
            self.downblocks.append(ResnetBlock3D(channels_b, channels_b, mode="default", channels_emb=channels_emb, compile=compile, use_checkpoint=use_checkpoint,
                                                 causal=causal, learnable_pool=learnable_pool))
            #self.skips.append()
            skip_channels.append(channels_b)
            if -1 in attn_res:
                self.downblocks.append(Attention3D(channels_b, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))


        self.bottleneck = nn.ModuleList()
        if self.use_injection and injection_res==-1:
            self.injection_proj = nn.Conv3d(channels_injection, channels_b, kernel_size=3, stride=1, padding=1, bias=True)
            self.bottleneck.append(ResnetBlock3D(channels_b*2, channels_b, mode="default", channels_emb=channels_emb, compile=compile, use_checkpoint=use_checkpoint,
                                                 causal=causal, learnable_pool=learnable_pool))
        else:
            self.bottleneck.append(ResnetBlock3D(channels_b, channels_b, mode="default", channels_emb=channels_emb, compile=compile, use_checkpoint=use_checkpoint,
                                                 causal=causal, learnable_pool=learnable_pool))
        if -1 in attn_res:
            self.bottleneck.append(Attention3D(channels_b, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))
        self.bottleneck.append(ResnetBlock3D(channels_b, channels_b, mode="default", channels_emb=channels_emb, compile=compile, use_checkpoint=use_checkpoint,
                                             causal=causal, learnable_pool=learnable_pool))


        self.upblocks = nn.ModuleList()
        for j in range(num_blocks+1):
            self.upblocks.append(ResnetBlock3D(channels_b+skip_channels.pop(), channels_b, mode="default", channels_emb=channels_emb, compile=compile, use_checkpoint=use_checkpoint,
                                               causal=causal, learnable_pool=learnable_pool))
            if -1 in attn_res:
                self.upblocks.append(Attention3D(channels_b, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))
        for i in reversed(range(len(self.channel_mult)-1)):
            channels_b = self.channel_mult[i]*channels
            channels_a = self.channel_mult[i+1]*channels

            if mode_res is None:
                decoder_mode = ""
            else:
                decoder_mode = "_"+mode_res[i] if mode_res[i] != "" else ""

            self.upblocks.append(ResnetBlock3D(channels_a, channels_b, mode="up"+decoder_mode, channels_emb=channels_emb, compile=compile, use_checkpoint=use_checkpoint,
                                               causal=causal, learnable_pool=learnable_pool))
            for j in range(num_blocks+1):
                self.upblocks.append(ResnetBlock3D(channels_b+skip_channels.pop(), channels_b, mode="default", channels_emb=channels_emb, compile=compile, use_checkpoint=use_checkpoint,
                                                   causal=causal, learnable_pool=learnable_pool))
                if i+1 in attn_res:
                    self.upblocks.append(Attention3D(channels_b, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, compile=compile, use_checkpoint=use_checkpoint))

        self.proj_out = nn.Sequential(nn.GroupNorm(32, channels), nn.SiLU(), zero_init(Conv3D(channels, channels_out, kernel_size=3, stride=1, padding=1, bias=True, causal=causal)))

    def forward_with_cfg(self, x, t, inj=None, guidance=1.0):
        pass

    def forward(self, x, t, inj=None):
        emb = self.timestep_embedder(t)
        emb = F.silu(emb)

        x = self.proj_in(x)

        injected = False if self.use_injection else True
        skips = [x]
        for block in self.downblocks:
            if not isinstance(block, Attention3D):
                #inject
                if hasattr(block, "mode") and not injected:
                    if block.mode=="default":
                        if x.size(-2)==inj.size(-2) and x.size(-1)==inj.size(-1):
                            x = torch.cat([x,self.injection_proj(inj)], dim=1)
                            injected = True
                x = block(x, emb)
                skips.append(x)
            else:
                x = block(x)

        if self.use_injection and not injected and self.injection_res==-1:
            x = torch.cat([x,self.injection_proj(inj)], dim=1)
            injected = True

        for block in self.bottleneck:
            if not isinstance(block, Attention3D):
                x = block(x, emb)
            else:
                x = block(x)

        for block in self.upblocks:
            if isinstance(block, Attention3D):
                x = block(x)
            elif hasattr(block, "mode"):
                if block.mode=="up":
                    x = block(x, emb)
                else:
                    x = torch.cat([x, skips.pop()], dim=1)
                    x = block(x, emb) 
        
        x = self.proj_out(x)

        return x
