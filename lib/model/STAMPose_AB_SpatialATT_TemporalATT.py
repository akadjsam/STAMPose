# Ablation: Spatial Attention + Temporal Attention (vs SATMPose: Spatial Attention + Temporal Mamba)
# Replace TTEblock: DCTSSMBlock -> TemporalAttentionBlock, rest of the structure unchanged
import math
from functools import partial

from einops import rearrange
import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_


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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: (B*F, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SpatialAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpatialAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        # x: (B*F, N, C)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: (B*N, F, C) — attention over frames
        B, F, C = x.shape
        qkv = self.qkv(x).reshape(B, F, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, F, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TemporalAttentionBlock(nn.Module):
    """
    Block that replaces DCTSSMBlock (Temporal Mamba).
    Keeps the input/output shape (b, f, n, c), so it can be used as a drop-in replacement in the same forward flow as SATMPose.
    Internally reshapes to (b*n, f, c) to perform attention over the frame dimension.
    """
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TemporalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        # x: (b, f, n, c)
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c -> (b n) f c')
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, '(b n) f c -> b f n c', n=n)
        return x


class STAMPose_AB_SA_TA(nn.Module):
    def __init__(self, num_heads=4, num_frame=243, num_joints=17, in_chans=2, embed_dim_ratio=128, depth=10,
                 mlp_ratio=2., drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """
        Ablation: Temporal Mamba -> Temporal Attention
        STE: SpatialAttentionBlock (unchanged)
        TTE: TemporalAttentionBlock (replaces Mamba)
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        out_dim = 3

        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            SpatialAttentionBlock(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.TTEblocks = nn.ModuleList([
            TemporalAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def STE_forward(self, x):
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = rearrange(x, '(b f) n c -> b f n c', f=f)
        x = self.Spatial_norm(x)
        return x

    def TTE_foward(self, x):
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c -> (b n) f c')
        x += self.Temporal_pos_embed[:, :f, :]
        x = self.pos_drop(x)
        x = rearrange(x, '(b n) f c -> b f n c', n=n)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape) == 4
        b, f, n, c = x.shape
        for i in range(1, self.block_depth):
            x = rearrange(x, 'b f n c -> (b f) n c')
            x = self.STEblocks[i](x)
            x = rearrange(x, '(b f) n c -> b f n c', f=f)
            x = self.Spatial_norm(x)

            x = self.TTEblocks[i](x)
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
