    
import os
import sys
import argparse
import pickle
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone


# ----------------------------
# Skeleton utilities
# ----------------------------
I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

JOINT_NAMES = [
    "Hip", "R Hip", "R Knee", "R Foot",
    "L Hip", "L Knee", "L Foot",
    "Spine", "Thorax", "Neck", "Head",
    "L Shoulder", "L Elbow", "L Wrist",
    "R Shoulder", "R Elbow", "R Wrist"
]


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    if torch.is_tensor(data):
        flipped = data.clone()
    else:
        flipped = copy.deepcopy(data)

    flipped[..., 0] *= -1
    flipped[..., left_joints + right_joints, :] = flipped[..., right_joints + left_joints, :]
    return flipped


def root_center(pose3d):
    pose3d = pose3d.copy()
    pose3d = pose3d - pose3d[0:1]
    return pose3d

def to_plot_coords(pose):
    """pose: [N, 3]  →  [N, 3]  (converted to a matplotlib-3D-friendly coordinate system)"""
    out = np.empty_like(pose)
    out[:, 0] = pose[:, 0]      # x
    out[:, 1] = pose[:, 2]      # depth → matplotlib y
    out[:, 2] = -pose[:, 1]     # -image_y → matplotlib z(up)
    return out


def set_axes_equal(ax, data):
    """data: [N, 3] (values already converted to the plot coordinate system)"""
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    x_mid = (x.max() + x.min()) / 2.0
    y_mid = (y.max() + y.min()) / 2.0
    z_mid = (z.max() + z.min()) / 2.0

    max_range = max(
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ) / 2.0

    if max_range < 1e-6:
        max_range = 1.0

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)


def draw_pose_pair(ax, pred, gt):
    """
    pred, gt: [17, 3] (original data coordinate system)
    GT  : gray line + orange joint dots
    Pred: blue line + blue joint dots
    """
    pred = root_center(pred)
    gt = root_center(gt)

    pred_p = to_plot_coords(pred)
    gt_p = to_plot_coords(gt)

    # ---------- colors ----------
    gt_line_color = (0.65, 0.65, 0.65)
    gt_dot_color  = (1.00, 0.55, 0.00)
    pr_line_color = (0.10, 0.30, 0.95)
    pr_dot_color  = (0.10, 0.30, 0.95)

    # ---------- draw GT skeleton ----------
    for i in range(len(I)):
        xs = [gt_p[I[i], 0], gt_p[J[i], 0]]
        ys = [gt_p[I[i], 1], gt_p[J[i], 1]]
        zs = [gt_p[I[i], 2], gt_p[J[i], 2]]
        ax.plot(xs, ys, zs, lw=2.6, color=gt_line_color, alpha=1.0)

    # ---------- draw Pred skeleton ----------
    for i in range(len(I)):
        xs = [pred_p[I[i], 0], pred_p[J[i], 0]]
        ys = [pred_p[I[i], 1], pred_p[J[i], 1]]
        zs = [pred_p[I[i], 2], pred_p[J[i], 2]]
        ax.plot(xs, ys, zs, lw=2.0, color=pr_line_color, alpha=1.0)

    # ---------- joint dots ----------
    ax.scatter(gt_p[:, 0], gt_p[:, 1], gt_p[:, 2],
               c=[gt_dot_color], s=30, depthshade=False)
    ax.scatter(pred_p[:, 0], pred_p[:, 1], pred_p[:, 2],
               c=[pr_dot_color], s=24, depthshade=False)

    # ---------- axis range ----------
    all_pts = np.concatenate([pred_p, gt_p], axis=0)
    set_axes_equal(ax, all_pts)

    # 3D box aspect (preserve human body proportions)
    ax.set_box_aspect((1, 1, 1))

    ax.view_init(elev=10., azim=-75)

    # ---------- pane / grid ----------
    pane_color = (1.0, 1.0, 1.0, 0.08)
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)

    ax.grid(True, alpha=0.45)

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.tick_params(axis='x', which='both', length=0, pad=0)
    ax.tick_params(axis='y', which='both', length=0, pad=0)
    ax.tick_params(axis='z', which='both', length=0, pad=0)


# ----------------------------
# PKL parsing
# ----------------------------
def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return np.array(x)


def squeeze_to_tfjd(arr):
    arr = to_numpy(arr)

    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    while arr.ndim > 3 and arr.shape[1] == 1:
        arr = arr[:, 0]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array [T, J, D], but got shape {arr.shape}")

    return arr


def find_first_key(d, candidates):
    for k in candidates:
        if k in d:
            return d[k], k
    return None, None


def print_structure(obj, prefix="root", max_depth=2, depth=0):
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        print(f"{prefix}: dict keys = {list(obj.keys())}")
        for k, v in obj.items():
            if isinstance(v, (dict, list, tuple)):
                print_structure(v, prefix=f"{prefix}.{k}", max_depth=max_depth, depth=depth+1)
            else:
                try:
                    arr = np.asarray(v)
                    print(f"{prefix}.{k}: shape={arr.shape}, dtype={arr.dtype}")
                except Exception:
                    print(f"{prefix}.{k}: type={type(v)}")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}: {type(obj).__name__}, len={len(obj)}")
        if len(obj) > 0:
            print_structure(obj[0], prefix=f"{prefix}[0]", max_depth=max_depth, depth=depth+1)
    else:
        try:
            arr = np.asarray(obj)
            print(f"{prefix}: shape={arr.shape}, dtype={arr.dtype}")
        except Exception:
            print(f"{prefix}: type={type(obj)}")


def extract_sample_from_pkl(pkl_obj, sample_idx=0):
    if isinstance(pkl_obj, (list, tuple)):
        return pkl_obj[sample_idx]

    if isinstance(pkl_obj, dict):
        for key in ["data", "samples", "annotations", "clips"]:
            if key in pkl_obj and isinstance(pkl_obj[key], (list, tuple)):
                return pkl_obj[key][sample_idx]
        return pkl_obj

    raise TypeError(f"Unsupported PKL top-level type: {type(pkl_obj)}")


def extract_input_and_gt(sample):
    if not isinstance(sample, dict):
        raise TypeError(f"Sample must be dict-like, got {type(sample)}")

    input_candidates = [
        "data_input", "input_2d", "keypoints_2d", "joints_2d",
        "pose_2d", "kp_2d", "input", "data_input_2d"
    ]
    gt_candidates = [
        "data_label", "gt_3d", "joints_3d", "pose_3d",
        "label", "target", "data_gt", "output_3d"
    ]

    input_2d, input_key = find_first_key(sample, input_candidates)
    gt_3d, gt_key = find_first_key(sample, gt_candidates)

    if input_2d is None or gt_3d is None:
        print("\n[DEBUG] Could not auto-find input/gt keys.")
        print_structure(sample)
        raise KeyError(
            f"Failed to find input_2d / gt_3d keys.\n"
            f"Input tried: {input_candidates}\n"
            f"GT tried: {gt_candidates}"
        )

    input_2d = squeeze_to_tfjd(input_2d)
    gt_3d = squeeze_to_tfjd(gt_3d)

    if gt_3d.shape[-1] > 3:
        gt_3d = gt_3d[..., :3]

    print(f"[INFO] input key: {input_key}, shape={input_2d.shape}")
    print(f"[INFO] gt key   : {gt_key}, shape={gt_3d.shape}")

    return input_2d, gt_3d


# ----------------------------
# Model inference
# ----------------------------
@torch.no_grad()
def infer_prediction(model, config, input_2d):
    x = input_2d.copy()

    if getattr(config, "no_conf", False):
        x = x[..., :2]

    x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).cuda()
    x_flip = flip_data(x)

    pred = model(x)
    pred_flip = flip_data(model(x_flip))
    pred = (pred + pred_flip) / 2.0

    pred = pred[0].detach().cpu().numpy()
    return pred


# ----------------------------
# Figure creation
# ----------------------------
def choose_frames(T, frames=None):
    if frames is not None and len(frames) > 0:
        return [min(max(0, f), T - 1) for f in frames]
    if T >= 3:
        return [0, T // 2, T - 1]
    return list(range(T))


def make_figure(pred_3d, gt_3d, frames, out_file):
    n = len(frames)
    fig = plt.figure(figsize=(4.5, 4.2 * n))

    for idx, f in enumerate(frames):
        ax = fig.add_subplot(n, 1, idx + 1, projection="3d")
        draw_pose_pair(ax, pred_3d[f], gt_3d[f])

    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.02,
        top=0.98,
        hspace=0.15
    )

    plt.savefig(out_file, dpi=300)
    plt.close(fig)

    print(f"[INFO] Saved figure to: {out_file}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pose3d/STAMPose_train_h36m_B.yaml",
        help="Path to config yaml"
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        default="STAMPose_train_h36m_B/best_epoch.bin",
        help="Path to checkpoint bin"
    )

    random_seed = random.randint(0, 2227)
    formatted_seed = f"{random_seed:04d}"
    print(f"Selected random seed: {formatted_seed}")
    parser.add_argument(
        "--pkl",
        type=str,
        default=f"data/motion3d/MB3D_f243s81/H36M-SH/test/0000{formatted_seed}.pkl",
        help="Path to one test pkl file"
    )

    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="If pkl contains a list of samples, choose one"
    )

    random_frames = random.sample(range(0, 243), 3)
    print(f"Selected random frames: {random_frames}")
    parser.add_argument(
        "--frames",
        type=int,
        nargs="*",
        default=None,
        help=f"Frame indices to visualize, e.g. --frames {random_frames[0]} {random_frames[1]} {random_frames[2]}"
    )

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--out", type=str, default="qualitative_overlay.png")
    parser.add_argument("--model_name", type=str, default="STAMPose")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load model
    config = get_config(args.config)
    model_backbone = load_backbone(config)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone).cuda()

    print("[INFO] Loading checkpoint:", args.evaluate)
    checkpoint = torch.load(args.evaluate, map_location=lambda storage, loc: storage)

    state_dict = checkpoint["model_pos"]
    model_backbone.load_state_dict(state_dict, strict=True)
    model_backbone.eval()

    # Load PKL
    print("[INFO] Loading PKL:", args.pkl)
    with open(args.pkl, "rb") as f:
        pkl_obj = pickle.load(f)

    sample = extract_sample_from_pkl(pkl_obj, sample_idx=args.sample_idx)
    input_2d, gt_3d = extract_input_and_gt(sample)

    # Inference
    pred_3d = infer_prediction(model_backbone, config, input_2d)

    # Match length if needed
    T = min(len(pred_3d), len(gt_3d))
    pred_3d = pred_3d[:T]
    gt_3d = gt_3d[:T]

    frames = choose_frames(T, args.frames)

    make_figure(pred_3d, gt_3d, frames, args.out)


if __name__ == "__main__":
    main()