import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x: torch.Tensor, casual_mask=False) -> torch.Tensor:
        # x: (Batch_size, seq_len, Height*Width)
        
        input_shape = x.shape
        
        batch_size, sequence_length, embed_dim = input_shape
        
        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        
        # x: (Batch_size, seq_len , Dim) -> 3*(Batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch_size, seq_len, Dim) -> (Batch_size, seq_len, d_heads, d_embed / d_head) -> (Batch_size, d_heads, seq_len, seq_len)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2) 
        

        weight = q @ k.transpose(-1, -2)
        
        # implement causal mask