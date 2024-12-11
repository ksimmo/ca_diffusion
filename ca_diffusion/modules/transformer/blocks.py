import torch
import torch.nn as nn

from ca_diffusion.modules.utils import checkpoint
from ca_diffusion.modules.norm import RMSNorm, AdaRMSNorm
from ca_diffusion.modules.transformer.attention import CrossAttention
from ca_diffusion.modules.transformer.feedforward import FeedForward

########################
#vanilla transformer blocks
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads, eps=1e-6, compile=True, **attention_kwargs):
        super().__init__()

        self.norm = RMSNorm(channels, eps=eps)
        self.attention = CrossAttention(channels, num_heads=num_heads, eps=eps, **attention_kwargs)

        if compile:
            self.forward = torch.compile(self.forward)

    def forward(self, x):
        h = self.attention(self.norm(x))
        return x+h
    
class FeedForwardBlock(nn.Module):
    def __init__(self, channels, eps=1e-6, compile=True, **ff_kwargs):
        super().__init__()

        self.norm = RMSNorm(channels, eps=eps)
        self.feedforward = FeedForward(channels, **ff_kwargs)

        if compile:
            self.forward = torch.compile(self.forward, fullgraph=True)

    def forward(self, x):
        h = self.feedforward(self.norm(x))
        return x+h
    

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, head_channels=None, qkv_bias=True, bias=True, qk_norm=True, attn_dropout=0.0, expand=4, ff_dropout=0.0, ff_act="default",
                 eps=1e-6, compile_attn=True, compile_ff=True, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.attn = AttentionBlock(channels, num_heads=num_heads, eps=eps, compile=compile_attn, head_channels=head_channels, qkv_bias=qkv_bias, bias=bias, qk_norm=qk_norm, dropout=attn_dropout)
        self.ff = FeedForwardBlock(channels, compile=compile_ff, expand=expand, bias=bias, dropout=ff_dropout, act=ff_act)

    def forward(self, x):
        x = checkpoint(self.attn, self.use_checkpoint, x)
        x = checkpoint(self.ff, self.use_checkpoint, x)
        return x
    
########################
#conditional transformer blocks -> useful for DiT
class ConditionAttentionBlock(nn.Module):
    def __init__(self, channels, emb_channels, num_heads, eps=1e-6, compile=True, **attention_kwargs):
        super().__init__()

        self.norm = AdaRMSNorm(channels, emb_channels, eps=eps)
        self.attention = CrossAttention(channels, num_heads=num_heads, eps=eps, **attention_kwargs)

        if compile:
            self.forward = torch.compile(self.forward)

    def forward(self, x, emb):
        h = self.attention(self.norm(x, emb))
        return x+h
    
class ConditionFeedForwardBlock(nn.Module):
    def __init__(self, channels, emb_channels, eps=1e-6, compile=True, **ff_kwargs):
        super().__init__()

        self.norm = AdaRMSNorm(channels, emb_channels, eps=eps)
        self.feedforward = FeedForward(channels, **ff_kwargs)

        if compile:
            self.forward = torch.compile(self.forward, fullgraph=True)

    def forward(self, x, emb):
        h = self.feedforward(self.norm(x, emb))
        return x+h
    
class ConditionTransformerBlock(nn.Module):
    def __init__(self, channels, emb_channels, num_heads, head_channels=None, qkv_bias=True, bias=True, qk_norm=True, attn_dropout=0.0, expand=4, ff_dropout=0.0, ff_act="default",
                 eps=1e-6, compile_attn=True, compile_ff=True, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.attn = ConditionAttentionBlock(channels, emb_channels, num_heads=num_heads, eps=eps, compile=compile_attn, head_channels=head_channels, 
                                            qkv_bias=qkv_bias, bias=bias, qk_norm=qk_norm, dropout=attn_dropout)
        self.ff = ConditionFeedForwardBlock(channels, emb_channels, compile=compile_ff, expand=expand, bias=bias, dropout=ff_dropout, act=ff_act)

    def forward(self, x, emb):
        x = checkpoint(self.attn, self.use_checkpoint, x, emb)
        x = checkpoint(self.ff, self.use_checkpoint, x, emb)
        return x



