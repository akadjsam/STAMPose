# Ablation: Spatial Attention → Spatial Mamba (PoseMamba forwardv2 style)
#           Temporal axis: SATMPose's DCTSSMBlock (v2_plus_poselimbs_t, K=2) — unchanged
#
# NOTE: This is the version BEFORE the temporal leakage fix.
#   forward_core receives (B, D_inner, H=F=243, W=N=17).
#   SpatialBidirectionalScan.flatten(2,3) produces a 4131-length sequence,
#   so SSM state flows across frame boundaries — NOT a pure spatial SSM.
#   See SATMPose_AB_SM.py for the fixed version (B*F merge → H=1).
import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath

from lib.model.mambablocks import DCTSSMBlock, mamba_init, DCTSSM_v2, Mlp

try:
    from lib.model.csms6s import (
        SpatialBidirectionalScan,
        SpatialBidirectionalMerge,
        SelectiveScanCore,
    )
except ImportError:
    from csms6s import (
        SpatialBidirectionalScan,
        SpatialBidirectionalMerge,
        SelectiveScanCore,
    )


class SpatialDCTSSM_before(nn.Module, mamba_init, DCTSSM_v2):
    """
    Spatial Mamba SSM — PRE-FIX version (temporal leakage present).
    forward_core receives (B, D, F=243, N=17): SpatialBidirectionalScan
    flattens to F*N=4131, causing SSM state to cross frame boundaries.
    """
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        channel_first=False,
        **kwargs,
    ):
        super().__init__()

        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1

        self.disable_force32 = False
        self.oact = False
        self.disable_z = False
        self.disable_z_act = False

        k_group = 2

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act = act_layer()

        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=(1, d_conv),
                padding=(0, (d_conv - 1) // 2),
            )

        _x_proj = [nn.Linear(d_inner, dt_rank + d_state * 2, bias=False) for _ in range(k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in _x_proj], dim=0))
        del _x_proj

        self.out_act = nn.Identity()
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.out_norm = nn.LayerNorm(d_inner)

        if initialize == "v0":
            _dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in _dt_projs], dim=0))
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in _dt_projs], dim=0))
            del _dt_projs

            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)

        self.forward_core = partial(
            self.forward_corev2,
            force_fp32=True,
            CrossScan=SpatialBidirectionalScan,
            SelectiveScan=SelectiveScanCore,
            CrossMerge=SpatialBidirectionalMerge,
        )

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Input:  (B, F, N, C)
        Output: (B, F, N, C)

        No B*F merge: forward_core receives (B, D_inner, H=F=243, W=N=17).
        SpatialBidirectionalScan.flatten(2,3) → sequence length = F*N = 4131.
        SSM state propagates across frame boundaries → temporal leakage.
        """
        b, f, n, _ = x.shape

        x = self.in_proj(x)                           # (B, F, N, 2*D_inner)
        x, z = x.chunk(2, dim=-1)                    # each (B, F, N, D_inner)
        z = self.act(z)

        # Permute to channel-first WITHOUT merging frames into batch
        x = x.permute(0, 3, 1, 2).contiguous()       # (B, D_inner, H=F=243, W=N=17)

        if self.with_dconv:
            x = self.conv2d(x)                        # (1×d_conv): H unchanged, W(joint) convolved
        x = self.act(x)

        # forward_corev2: (B, D, F=243, N=17) — flatten → 4131-length sequence
        y = self.forward_core(x)                      # (B, F, N, D_inner)
        y = self.out_act(y)

        y = y * z                                     # element-wise gate

        return self.dropout(self.out_proj(y))         # (B, F, N, D_model)


class SpatialDCTSSMBlock_before(nn.Module):
    """Pre-norm wrapper for SpatialDCTSSM_before."""
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
        ssm_ratio=2.0,
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        mlp_ratio=2.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.op = SpatialDCTSSM_before(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(hidden_dim)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            act_layer=mlp_act_layer,
            drop=mlp_drop_rate,
        )

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class STAMPose_SS_TS(nn.Module):
    def __init__(
        self,
        num_frame=243,
        num_joints=17,
        in_chans=2,
        embed_dim_ratio=128,
        depth=10,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=None,
        **kwargs,
    ):
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
            SpatialDCTSSMBlock_before(
                hidden_dim=embed_dim_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                ssm_ratio=2.0,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])

        self.TTEblocks = nn.ModuleList([
            DCTSSMBlock(
                hidden_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                forward_type='v2_plus_poselimbs_t',
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
        x = rearrange(x, '(b f) n c -> b f n c', f=f)

        x = self.STEblocks[0](x)
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
        for i in range(1, self.block_depth):
            x = self.STEblocks[i](x)
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
        return x.view(b, f, n, -1)