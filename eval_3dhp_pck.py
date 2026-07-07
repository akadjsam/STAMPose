
import argparse
import numpy as np
import scipy.io as scio

# ---- protocol constants (official 3DHP test) -----------------------------
ROOT_IDX = 14                       # pelvis index in the 17-joint test order
PCK_THRESHOLD = 150.0               # mm
AUC_THRESHOLDS = np.linspace(0.0, 150.0, 31)   # 0,5,...,150 mm  (31 values)
JOINT_PERM = None                   # set to a list of 17 indices to reorder GT->pred order
# --------------------------------------------------------------------------


def _to_TxJx3(arr):
    """Normalize an array of 51 coords/frame to shape (T, 17, 3) regardless of layout."""
    a = np.asarray(arr).squeeze()
    # collapse a possible singleton (the (3,17,1,T) -> (3,17,T) etc. already squeezed)
    a = np.squeeze(a)
    if a.ndim == 2:
        # (T, 51) or (51, T)
        if a.shape[1] == 51:
            return a.reshape(a.shape[0], 17, 3)
        if a.shape[0] == 51:
            return a.T.reshape(a.shape[1], 17, 3)
        raise ValueError(f"Unexpected 2D shape {a.shape}")
    if a.ndim != 3:
        raise ValueError(f"Unexpected ndim {a.ndim} shape {a.shape}")
    # find the axis of size 3 and the axis of size 17; the remaining axis is time
    dims = list(a.shape)
    ax3 = dims.index(3)
    ax17 = dims.index(17) if 17 in dims else [i for i in range(3) if i != ax3][0]
    axt = [i for i in range(3) if i not in (ax3, ax17)][0]
    return np.transpose(a, (axt, ax17, ax3))


def root_relative(x, root=ROOT_IDX):
    return x - x[:, root:root + 1, :]


def per_joint_error(pred, gt):
    """pred, gt: (T,17,3) -> (T,17) euclidean distance in mm."""
    return np.linalg.norm(pred - gt, axis=-1)


def compute_metrics(err):
    """err: (Nframes, 17). Returns (mpjpe, pck@150, auc)."""
    mpjpe = err.mean()
    pck = (err < PCK_THRESHOLD).mean() * 100.0
    pck_curve = [(err < t).mean() for t in AUC_THRESHOLDS]
    auc = np.mean(pck_curve) * 100.0
    return mpjpe, pck, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='inference_data(_best).mat path')
    ap.add_argument('--gt', default='data_test_3dhp.npz', help='data_test_3dhp.npz path')
    args = ap.parse_args()

    pred_mat = scio.loadmat(args.pred)
    gt_all = np.load(args.gt, allow_pickle=True)['data'].item()

    seqs = sorted(k for k in pred_mat.keys() if not k.startswith('__'))
    print(f"sequences in prediction file: {seqs}\n")

    all_err = []
    print(f"{'seq':>6} | {'frames':>7} | {'MPJPE(mm)':>10} | {'PCK@150':>8} | {'AUC':>7}")
    print("-" * 52)

    for seq in seqs:
        if seq not in gt_all:
            print(f"{seq:>6} |  SKIP (no GT for this sequence in npz)")
            continue

        pred = _to_TxJx3(pred_mat[seq])                       # (Tp,17,3)
        gt = _to_TxJx3(gt_all[seq]['data_3d'])                # (Tg,17,3)
        valid = np.asarray(gt_all[seq]['valid']).squeeze().astype(bool)

        # predictions exist only for valid frames -> select valid GT frames
        gt = gt[valid]
        Tg = gt.shape[0]

        if JOINT_PERM is not None:
            gt = gt[:, JOINT_PERM, :]

        # defensive length alignment
        T = min(pred.shape[0], Tg)
        if pred.shape[0] != Tg:
            print(f"  [warn] {seq}: pred T={pred.shape[0]} vs valid GT T={Tg}; "
                  f"truncating to {T}")
        pred, gt = pred[:T], gt[:T]

        # both root-relative at pelvis
        pred = root_relative(pred)
        gt = root_relative(gt)

        err = per_joint_error(pred, gt)                        # (T,17)
        all_err.append(err)
        mpjpe, pck, auc = compute_metrics(err)
        print(f"{seq:>6} | {T:>7d} | {mpjpe:>10.2f} | {pck:>7.2f}% | {auc:>6.2f}%")

    err_cat = np.concatenate(all_err, axis=0)
    mpjpe, pck, auc = compute_metrics(err_cat)
    print("-" * 52)
    print(f"{'ALL':>6} | {err_cat.shape[0]:>7d} | {mpjpe:>10.2f} | {pck:>7.2f}% | {auc:>6.2f}%")
    print()
    print(f">> Overall MPJPE (P1): {mpjpe:.2f} mm   "
          f"(must match the P1 your training printed; if not, fix ROOT_IDX / JOINT_PERM)")
    print(f">> Overall PCK@150mm : {pck:.2f} %")
    print(f">> Overall AUC       : {auc:.2f} %")


if __name__ == '__main__':
    main()
