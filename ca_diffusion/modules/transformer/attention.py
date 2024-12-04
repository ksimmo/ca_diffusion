import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from ca_diffusion.modules.utils import zero_init,apply_wd
from ca_diffusion.modules.norm import RMSNorm

class CrossAttention(nn.Module):
    def __init__(self, channels, num_heads, head_channels=None, context_channels=None, qkv_bias=True, bias=True, qk_norm=True, dropout=0.0, eps=1e-6):
        super().__init__()

        self.num_heads = num_heads
        head_channels = channels//num_heads if head_channels is None else head_channels
        inner_channels = num_heads*head_channels
        context_channels = channels if context_channels is None else context_channels

        self.query = apply_wd(nn.Linear(channels, inner_channels, bias=qkv_bias))
        self.key = apply_wd(nn.Linear(context_channels, inner_channels, bias=qkv_bias))
        self.value = apply_wd(nn.Linear(context_channels, inner_channels, bias=qkv_bias))
        self.out = apply_wd(zero_init(nn.Linear(inner_channels, channels, bias=bias)))

        self.query_norm = RMSNorm(inner_channels, eps=eps) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(inner_channels, eps=eps) if qk_norm else nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        context = x if context is None else context
        query = self.query(x)
        key = self.key(context)
        value = self.value(context)

        query = self.query_norm(query)
        key = self.key_norm(key)

        query = rearrange(query, "... n (h c) -> ... h n c", h=self.num_heads)
        key = rearrange(key, "... n (h c) -> ... h n c", h=self.num_heads)
        value = rearrange(value, "... n (h c) -> ... h n c", h=self.num_heads)

        x = F.scaled_dot_product_attention(query, key, value)

        x = rearrange(x, "... h n c -> ... n (h c)")
        x = self.dropout(x)
        return self.out(x)