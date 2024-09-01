import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
        
        def __init__(self, in_channels: int):
            super().__init__()
            self.group_norm = nn.GroupNorm(32, in_channels)
            self.self_attention = SelfAttention(in_channels)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (Batch_size, in_channels, Height, Width)
            
           residue = x
           
           n, c, h, w = x.shape
           
           # x: (Batch_size, in_channels, Height, Width) -> (Batch_size, in_channels, Height*Width)
           x = x.view(n, c, h*w)
           
           # x: (Batch_size, in_channels, Height*Width) -> (Batch_size, Height*Width, in_channels)
           x = x.transpose(-1, -2)
           
           # x: (Batch_size, Height*Width, in_channels) -> (Batch_size, Height*Width, in_channels)
           x = self.self_attention(x)
           
           # x: (Batch_size, Height*Width, in_channels) -> (Batch_size, in_channels, Height*Width)
           x = x.transpose(-1, -2)
           
           # x: (Batch_size, in_channels, Height*Width) -> (Batch_size, in_channels, Height, Width)
           x = x.view((n,c,h,w))
           
           x += residue
           
           return x
           

class VAE_ResidualBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, in_channels, Height, Width)
        
        residue = x
        
        x = self.group_norm1(x)
        
        x = F.silu(x)
        
        x = self.conv1(x)
        
        x = self.group_norm2(x)
        
        x = F.silu(x)
        
        x = self.conv_2(x)
        
        return x + self.residual_layer(residue)
       