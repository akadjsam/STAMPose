# STAMPose

<p align="center">
  <b>STAMPose PyTorch implementation</b> · ACCV2026 · Spatial-Temporal Attention-Mamba for Efficient 3D Human Pose Estimation
</p>

<p align="center">
  <img src="sample.gif" width="70%" alt="STAMPose demo video" />
</p>

<p align="center">
  Video source is from <a href="https://www.youtube.com/shorts/QrGqxlrgyOc">Youtuber Nikoork</a>
</p>

---

## Introduction

Monocular 3D human pose estimation (3D HPE) from video sequences has significantly benefited from Transformers, but the quadratic complexity of the attention mechanism makes processing long sequences computationally expensive. While State Space Models (SSMs) with linear complexity offer an efficient alternative, purely SSM-based architectures flatten the joints into a fixed order, struggling to naturally handle the non-Euclidean spatial topology of 3D skeletal data.

**STAMPose** (Spatial-Temporal Attention-Mamba Pose estimation) addresses this limitation by proposing a hybrid architecture that assigns different operators to the spatial and temporal axes. Through axis-wise operator analysis, we found that the two axes are governed by opposite criteria:

* **Spatial Axis ($J=17$)**: On the short spatial axis, attention and SSM are equally accurate, so attention is chosen to maximize parallelization efficiency.
* **Temporal Axis ($T=243$)**: On the long temporal axis, attention is faster but less accurate, so a selective-scan SSM is chosen to prioritize accuracy. STAMPose deploys the **DC-TSSM (Dual-branch Concatenation Temporal SSM) block**, which fuses a bidirectional selective scan with a depthwise-convolution branch via channel concatenation.

By strategically deploying Attention for spatial modeling and DC-TSSM for temporal modeling, STAMPose matches or surpasses prior SSM-based models while achieving **1.7-2.0x higher throughput** at comparable parameter counts.

**paper** : [ACCV 2026](link1) · [arXiv](link2) · [PDF](link3)

## Results (Human3.6M)
Performance measured on a single NVIDIA RTX 3090Ti GPU.

| Model | Params | MACs/frame | P1 / GT (mm) | FPS |
|-------|--------|------------|--------------|-----|
| STAMPose-S | 0.8M | 14M | 40.8 / 18.6 | 41,675 |
| STAMPose-B | 3.3M | 57M | 38.7 / 14.0 | 25,058 |
| STAMPose-L | 6.7M | 113M | 37.9 / 12.3 | 12,469 |

## Installation

Environments : Python 3.8.5, PyTorch 2.4.1+cu121, CUDA 12.1.

```bash
git clone https://github.com/[YourUsername]/STAMPose.git
cd STAMPose

conda create -n STAMPose python=3.8.5
conda activate STAMPose
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cd kernels/selective_scan && pip install -e . && cd ../..
```

## Data Preparation
### Human 3.6M
1. Download MotionBERT preprocessed H3.6M data ([OneDrive](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ))
2. unzip it to `data/motion3d/`
   
```bash
cd tools && python convert_h36m.py && cd ..
```

### MPI-INF-3DHP
1. Please follow the dataset setup from [P-STMO](https://github.com/paTRICK-swk/P-STMO).
2. After preprocessing, the generated .npz files (data_train_3dhp.npz and data_test_3dhp.npz) should be located at data/motion3d directory.


## Training
### Human3.6M
1. Enter the path to the config file for `<configs>`. (e.g., configs/pose3d/STAMPose_train_h36m_S.yaml)
2. Specify the name of the checkpoint to save for `<checkpoint>`. (e.g., checkpoint/STAMPose_S)
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config <configs> --checkpoint <checkpoint>
```

### MPI-INF-3DHP
1. Enter the path to the config file for `configs`. (e.g., configs/MPI/STAMPose_mpi_81.yaml)
2. Specify the checkpoint directory to save for `checkpoint`. (e.g., checkpoint_mpi/STAMPose_81)
```bash
python train_3dhp.py --config <configs> --checkpoint <checkpoint>
```

## Evaluation
### Human3.6M
1. Enter the path to the trained model's config file for `configs`. (e.g., configs/pose3d/STAMPose_train_h36m_S.yaml)
2. Enter a title to save the evaluation results for `checkpoint_dir`. (e.g., eval)
3. Enter the path to the trained model for `checkpoint`. (e.g., STAMPose_S/best_epoch.bin)
```bash
python train.py --config <configs> -c <checkpoint_dir> -e <checkpoint>
```

### MPI-INF-3DHP
1. Specify `configs` and `checkpoint_dir` the same as during training, and enter the checkpoint filename to evaluate for `checkpoint_file`.
```bash
python train_3dhp.py --config <configs> --checkpoint <checkpoint_dir> --checkpoint-file <checkpoint_file> --eval-only
```

Pretrained weights can be downloaded from the link below.
[Download](https://drive.google.com/drive/folders/1m47HUeP5tkviIZmdtTI1gHJXJmslzjp4?usp=sharing)

## Demo
1. Download YOLOv3 + HRNet weights → ./demo/lib/checkpoint/ [Google Drive](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA)
2. Place the sample video in `demo/video`.
```
python vis.py --video sample.mp4 --gpu 0
```

## Project structure
```
STAMPose/
├── configs/
│   ├── pose3d/         # Human3.6M training configs (S / B / L)
│   ├── H36M_2D_GT/     # Human3.6M configs (2D ground-truth input)
│   └── MPI/            # MPI-INF-3DHP training configs
├── data/motion3d/      # Preprocessed H3.6M / MPI-INF-3DHP data (see Data Preparation)
├── lib/model/          # STAMPose backbone
├── kernels/selective_scan/  # CUDA SSM kernel (required)
├── tools/              # Dataset preprocessing
├── weights/            # Downloaded / trained checkpoints
├── vis.py              # In-the-wild demo
├── train.py            # Train & evaluate on Human3.6M
└── train_3dhp.py       # Train & evaluate on MPI-INF-3DHP
```

## Acknowledgement
Our code is based on the following repositories. We thank the authors for releasing the codes.
- [PoseMamba](https://github.com/nankingjing/PoseMamba)
- [SasMamba](https://github.com/HuCui2022/sasmamba_pose_estimation)
- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [MotionAGFormer](https://github.com/taatiteam/MotionAGFormer)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [VMamba](https://github.com/mzeromiko/vmamba)
- [MambaVision](https://github.com/nvlabs/mambavision)

## Citation
If you find this code or our paper useful, please cite it using the following format:

```
@inproceedings{stampose2026,
  title={STAMPose: Spatial-Temporal Attention-Mamba for Efficient 3D Human Pose Estimation},
  author={Anonymous},
  booktitle={ACCV 2026 Submission},
  year={2026}
}
```

## License
This project is released under the [Apache License 2.0](LICENSE).

## Contact
akadjsam@inha.edu or akadjsam@gmail.com