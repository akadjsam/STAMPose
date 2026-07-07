import torch

# pytorch cross scan =============
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)
class CrossScan_fs_ft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = xs[:, 0:2]
        # print('CrossScan_fs_ft')
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)
class CrossMerge_fs_ft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = xs[:, 0:2]
        xs = xs.view(B, 4, C, H, W)
        return xs
class CrossScan_bs_ft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = torch.flip(x, dims=[-1]).flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = xs[:, 0:2]
        # print('CrossScan_fs_ft')
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].view(B, 2, -1, L)
        y = ys[:, 0].view(B, -1, H, W).flip(dims=[-1]).contiguous().view(B, -1, L) + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)
class CrossMerge_bs_ft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].view(B, 2, D, -1)
        y = ys[:, 0].view(B, -1, H, W).flip(dims=[-1]).contiguous().view(B, D, -1) + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = torch.flip(x.view(B, C, H, W), dims=[-1]).flatten(2, 3)
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = xs[:, 0:2]
        xs = xs.view(B, 4, C, H, W)
        return xs
class CrossScan_fs_bt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = torch.flip(x.transpose(dim0=2, dim1=3), dims=[-1]).flatten(2, 3)
        xs[:, 2:4] = xs[:, 0:2]
        # print('CrossScan_fs_ft')
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).flip(dims=[-1]).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)
class CrossMerge_fs_bt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).flip(dims=[-1]).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = torch.flip(x.view(B, C, H, W).transpose(dim0=2, dim1=3), dims=[-1]).flatten(2, 3)
        xs[:, 2:4] = xs[:, 0:2]
        xs = xs.view(B, 4, C, H, W)
        return xs

class CrossScan_plus_poselimbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        assert W == 17, 'the number of joints is not 17'
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        indices = [0, 0, 1, 2, 3, 0, 4, 5, 6, 8, 11, 12, 13, 8, 14, 15, 16]
        
        # Scan 0: combine current joint + parent joint information
        xs[:, 0] = (x + x[..., indices]).flatten(2, 3)
        # Scan 1: axis-transposed scan
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        # Scans 2, 3: reverse-direction scans
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        
        # 1. Sum gradients from scan 0 (Forward) and scan 2 (Backward)
        grad_0 = (ys[:, 0] + ys[:, 2].flip(dims=[-1])).view(B, C, H, W)

        # 2. Sum gradients from scan 1 (Transpose) and scan 3 (Transpose+Backward), then restore axes
        grad_1 = (ys[:, 1] + ys[:, 3].flip(dims=[-1])).view(B, C, W, H).transpose(dim0=2, dim1=3)

        # 3. Sum the 1D gradients (error signal flowing into the current joint itself)
        grad_x = grad_0 + grad_1

        # 4. Accumulate gradients flowing to parent joints (indices)
        indices = torch.tensor([0, 0, 1, 2, 3, 0, 4, 5, 6, 8, 11, 12, 13, 8, 14, 15, 16], device=ys.device)
        grad_x.index_add_(dim=-1, index=indices, source=grad_0)
        
        return grad_x

class CrossMerge_plus_poselimbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # x is the gradient passed from the upper layer (grad_output)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        
        # Since the forward pass simply summed the 4 scan results,
        # the backward pass only needs to copy the gradient back into shape in the reverse direction
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs
    
# Performs only the temporal bidirectional scan
class TemporalBidirectionalScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        # x shape: (B, C, H, W) -> H is frames (T), W is joints (J)
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        
        # K=2 (bidirectional: Forward and Backward along the time axis)
        xs = x.new_empty((B, 2, C, H * W))

        # Key idea: transpose axes so that the time (H) flow is scanned as an independent sequence per joint (W)
        # (B, C, H, W) -> (B, C, W, H)
        x_time_seq = x.transpose(dim0=2, dim1=3)

        # 1. Forward Time Scan (past -> future)
        xs[:, 0] = x_time_seq.flatten(2, 3)

        # 2. Backward Time Scan (future -> past)
        xs[:, 1] = torch.flip(x_time_seq, dims=[-1]).flatten(2, 3)
        
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        
        # Sum the Forward and Backward gradients
        y0 = ys[:, 0].view(B, C, W, H)
        y1 = torch.flip(ys[:, 1].view(B, C, W, H), dims=[-1])
        
        y = y0 + y1
        
        # Restore to the original shape (B, C, H, W) and return
        return y.transpose(dim0=2, dim1=3)


class TemporalBidirectionalMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        # ys shape: (B, 2, C, H, W)
        B, K, C, H, W = ys.shape
        
        # ys contains a sequence internally flattened as H*W, so unflatten it back to 1D.
        ys_flat = ys.view(B, K, C, -1)

        # Restore to the original scanned shape (joint, frame) = (W, H).
        y0 = ys_flat[:, 0].view(B, C, W, H)
        y1 = torch.flip(ys_flat[:, 1].view(B, C, W, H), dims=[-1])
        
        y = y0 + y1
        
        # Finally, transpose back to the original shape (B, C, H, W) and return.
        return y.transpose(dim0=2, dim1=3).contiguous()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output shape: (B, C, H, W) -> received correctly as a 4D tensor.
        B, C, H, W = grad_output.shape
        
        # Create an empty gradient tensor matching the original ys shape (B, 2, C, H, W).
        grad_ys = grad_output.new_empty((B, 2, C, H, W))
        
        # Transpose axes to (B, C, W, H) for the time-axis computation.
        grad_y_time_seq = grad_output.transpose(dim0=2, dim1=3)
        
        # Distribute gradients in the reverse order, perfectly symmetric to the forward scan.
        grad_ys[:, 0] = grad_y_time_seq.reshape(B, C, H, W)
        grad_ys[:, 1] = torch.flip(grad_y_time_seq, dims=[-1]).reshape(B, C, H, W)
        
        return grad_ys


# Spatial bidirectional scan: joints 0→N (forward) and N→0 (backward) per frame, independently
class SpatialBidirectionalScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        # x: (B, C, H=F, W=N)  — H=frames, W=joints
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)

        xs = x.new_empty((B, 2, C, H * W))

        # Forward: scan joints 0→N for each frame (frame-major order)
        xs[:, 0] = x.flatten(2, 3)

        # Backward: scan joints N→0 for each frame (flip W axis, then frame-major)
        xs[:, 1] = torch.flip(x, dims=[-1]).flatten(2, 3)

        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape

        grad_0 = ys[:, 0].view(B, C, H, W)
        grad_1 = torch.flip(ys[:, 1].view(B, C, H, W), dims=[-1])  # un-flip W

        return grad_0 + grad_1


class SpatialBidirectionalMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        # ys: (B, 2, C, H, W)
        B, K, C, H, W = ys.shape
        ctx.shape = (H, W)

        ys_flat = ys.view(B, K, C, -1)  # (B, 2, C, H*W)

        y0 = ys_flat[:, 0].view(B, C, H, W)
        # Backward scan was over flip(x, W); restore by flipping back
        y1 = torch.flip(ys_flat[:, 1].view(B, C, H, W), dims=[-1])

        return (y0 + y1).contiguous()  # (B, C, H, W)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        H, W = ctx.shape
        B, C, _, _ = grad_output.shape

        grad_ys = grad_output.new_empty((B, 2, C, H, W))
        grad_ys[:, 0] = grad_output
        grad_ys[:, 1] = torch.flip(grad_output, dims=[-1])  # chain-rule through W-flip

        return grad_ys


class CrossScan_bs_bt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = torch.flip(x, dims=[-1]).flatten(2, 3)
        xs[:, 1] = torch.flip(x.transpose(dim0=2, dim1=3), dims=[-1]).flatten(2, 3)
        xs[:, 2:4] = xs[:, 0:2]
        # print('CrossScan_fs_ft')
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].view(B, 2, -1, L)
        y = ys[:, 0].view(B, -1, H, W).flip(dims=[-1]).contiguous().view(B, -1, L) + ys[:, 1].view(B, -1, W, H).flip(dims=[-1]).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)
class CrossMerge_bs_bt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].view(B, 2, D, -1)
        y = ys[:, 0].view(B, -1, H, W).flip(dims=[-1]).contiguous().view(B, D, -1) + ys[:, 1].view(B, -1, W, H).flip(dims=[-1]).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = torch.flip(x.view(B, C, H, W), dims=[-1]).flatten(2, 3)
        xs[:, 1] = torch.flip(x.view(B, C, H, W).transpose(dim0=2, dim1=3), dims=[-1]).flatten(2, 3)
        xs[:, 2:4] = xs[:, 0:2]
        xs = xs.view(B, 4, C, H, W)
        return xs
class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs

# these are for ablations =============
class CrossScan_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        return ys.sum(1).view(B, -1, H, W)


class CrossMerge_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        return ys.contiguous().sum(1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x.view(B, 4, C, H, W)


class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        return x
    
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        return ys.view(B, 4, -1, H, W).sum(1)


class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, H, W = ys.shape
        ctx.shape = (B, C, H, W)
        return ys.view(B, 4, -1, H * W).sum(1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, C, H, W = ctx.shape
        return x.view(B, 1, C, H, W).repeat(1, 4, 1, 1, 1)


# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


def check_nan_inf(tag: str, x: torch.Tensor, enable=True):
    if enable:
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(tag, torch.isinf(x).any(), torch.isnan(x).any(), flush=True)
            import pdb; pdb.set_trace()


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
  
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L  
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

# cross selective scan ===============================
# comment all checks if inside cross_selective_scan
class SelectiveScanMamba(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops




