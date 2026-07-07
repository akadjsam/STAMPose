# Structure: ATT-STE-ATT-STE repeated
import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt
import os
import sys

current_directory = os.path.dirname(__file__) + '/../' + '../'
sys.path.append(current_directory)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from functools import partial
import torch.fft

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np

from lib.model.mambablocks import DCTSSMBlock

import math

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_joints=17, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.num_joints = num_joints
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x shape: (B_total, N, C) where B_total = Batch * Frames
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Spatial Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SpatialAttentionBlock(nn.Module):
    """
    Spatial-Attention-based block that replaces MambaBlock (DCTSSMBlock)
    Structure: Norm -> Attention -> Residual -> Norm -> MLP -> Residual
    """
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpatialAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        # x: (B*F, N, C)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class STAMPose(nn.Module):
    def __init__(self,num_heads= 4,num_frame=243, num_joints=17, in_chans=2, embed_dim_ratio=128, depth=10, mlp_ratio=2., drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """
        Modified PoseMamba with Spatial Attention and Temporal Mamba
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3     #### output dimension is num_joints * 3
        
        # Embedding layers
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        # STEblocks (Spatial): replaced DCTSSMBlock -> SpatialAttentionBlock

        self.STEblocks = nn.ModuleList([
            SpatialAttentionBlock(
                dim=embed_dim_ratio,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=0.,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # TTEblocks (Temporal): original DCTSSMBlock (Mamba)
     
        self.TTEblocks = nn.ModuleList([
           DCTSSMBlock(
                hidden_dim = embed_dim, 
                mlp_ratio = mlp_ratio,
                drop_path=dpr[i], 
                norm_layer=norm_layer,
                forward_type='v2_plus_poselimbs_t' # set to Temporal Mamba
                )
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )

    def STE_forward(self, x):
        # Input: (b, f, n, c)
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c -> (b f) n c') # merge for Spatial Attention
        
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        
        # Run the first block (Attention)
        blk = self.STEblocks[0]
        x = blk(x) # SpatialAttentionBlock takes (BF, N, C) input and processes it

        # Restore to the original shape
        x = rearrange(x, '(b f) n c -> b f n c', f=f)

        # Pass through Spatial_norm once, following PoseMamba's structure (kept even though the block also has internal Norm)
        x = self.Spatial_norm(x)
        return x

    def TTE_foward(self, x):
        b, f, n, c  = x.shape
        # Convert to (B*N, F, C) shape for Temporal Mamba
        x = rearrange(x, 'b f n cw -> (b n) f cw', f=f)
        x += self.Temporal_pos_embed[:,:f,:]
        x = self.pos_drop(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        
        blk = self.TTEblocks[0]
        x = blk(x)

        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        
        # Repeat Spatial(Attention) -> Temporal(Mamba) for the given depth
        for i in range(1, self.block_depth):
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            # [Spatial Attention Step]
            # Reshape for Attention input: (B, F, N, C) -> (B*F, N, C)
            x = rearrange(x, 'b f n c -> (b f) n c')
            x = steblock(x)
            x = rearrange(x, '(b f) n c -> b f n c', f=f)
            x = self.Spatial_norm(x)
            
            # [Temporal Mamba Step]
            # DCTSSMBlock is internally designed to process (B, F, N, C) input (based on the original PoseMamba)
            # Internal behavior may vary depending on the TTEBlock setting, but the original ST_forward flow is kept
            x = tteblock(x)
            x = self.Temporal_norm(x)
            
        return x

    def forward(self, x):
        b, f, n, c = x.shape
        x = self.STE_forward(x)
        x = self.TTE_foward(x)
        x = self.ST_foward(x)
        x = self.head(x)
        x = x.view(b, f, n, -1)
        return x

if __name__ == "__main__":
    # Test code
    try:
        torch.cuda.set_device(0)
        device = 'cuda'
    except:
        device = 'cpu'
        
    model = SATMPose(num_heads=4, num_frame=243, embed_dim_ratio=128, mlp_ratio=2, depth=10, drop_path_rate=0.2).to(device)
    
    # Input test: (Batch, Frames, Joints, Channels)
    from thop import profile, clever_format
    input_shape = (1, 243, 17, 2)
    x = torch.randn(input_shape).cuda()
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" %(flops))
    print("params: %s" %(params))