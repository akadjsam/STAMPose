import torch
from thop import profile
from lib.utils.learning import load_backbone
from lib.utils.tools import get_config

WARMUP = 10
N = 100
B = 32  # reduce if OOM occurs

def measure_all_metrics(config_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    args = get_config(config_path)
    model = load_backbone(args)
    model.to(device)

    input_b1 = torch.randn(1, 243, 17, 2, device=device)
    input_bN = torch.randn(B, 243, 17, 2, device=device)

    # ── MACs / Params (based on batch=1) ────────────────────────────
    model.eval()
    macs, params = profile(model, inputs=(input_b1,), verbose=False)
    macs_g         = macs / 1e9
    params_m       = params / 1e6
    macs_per_frame = (macs / 243) / 1e6

    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)

    # ── Inference Latency (B=1) ──────────────────────────────────────
    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            model(input_b1)
    torch.cuda.synchronize()

    starter.record()
    with torch.no_grad():
        for _ in range(N):
            model(input_b1)
    ender.record()
    torch.cuda.synchronize()
    infer_latency_ms = starter.elapsed_time(ender) / N

    # ── Inference Throughput (B=B) ────────────────────────────────────
    with torch.no_grad():
        for _ in range(WARMUP):
            model(input_bN)
    torch.cuda.synchronize()

    starter.record()
    with torch.no_grad():
        for _ in range(N):
            model(input_bN)
    ender.record()
    torch.cuda.synchronize()
    infer_thru_ms = starter.elapsed_time(ender) / N
    infer_fps     = (1000.0 / infer_thru_ms) * (B * 243)

    # ── Output ────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("         MODEL EFFICIENCY METRICS (FP32)        ")
    print("=" * 50)
    print(f"Params (M)                    : {params_m:.4f} M")
    print(f"MACs (G)                      : {macs_g:.4f} G")
    print(f"MACs/frame (M)                : {macs_per_frame:.4f} M")
    print("-" * 50)
    print(f"Inference latency  (B=1)      : {infer_latency_ms:.2f} ms  (N={N})")
    print(f"Inference throughput (B={B})  : {infer_fps:.1f} FPS  (N={N})")
    print("=" * 50)


if __name__ == "__main__":
    configs = [
        "configs/pose3d/STAMPose_train_h36m_S.yaml",
        "configs/pose3d/STAMPose_train_h36m_B.yaml",
        "configs/pose3d/STAMPose_train_h36m_L.yaml",
    ]
    for cfg in configs:
        print(f"\n>>> {cfg}")
        measure_all_metrics(cfg)
