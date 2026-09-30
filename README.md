# STAMPose

<p align="center">
  <b>STAMPose PyTorch implementation</b> · ACCV2026 · Spatial-Temporal Attention-Mamba for Efficient 3D Human Pose Estimation
</p>

<p align="center">
  <a href="https://github.com/akadjsam/STAMPose">
    <img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://pytorch.org">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <img src="https://img.shields.io/badge/ACCV-2026-3776AB" alt="Conference">
  <!-- <a href="논문_링크">
    <img src="https://img.shields.io/badge/arXiv-0000.00000-B31B1B" alt="arXiv">
  </a> -->
  <a href="https://github.com/akadjsam/STAMPose/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-7E9E35" alt="License">
  </a>
</p>
  
<p align="center">
  <video src="demo/output/BTS-2.0.mp4" width="70%" alt="STAMPose demo video (BTS - 2.0)" />
</p>

<p align="center">
  Video source is from <a href="https://www.youtube.com/shorts/QrGqxlrgyOc">Youtuber Nikoork</a>
</p>

<p align="center">
  <video src="demo/output/cortis_redred.mp4" width="70%" alt="STAMPose demo video (cortis - REDRED)" />
</p>

<p align="center">
  Video source is from <a href="https://www.youtube.com/watch?v=tNEpSOfsA3Y&list=LL&index=39">Youtuber Nikoork</a>
</p>

---

## Introduction

Monocular 3D human pose estimation (3D HPE) from video has benefited significantly from Transformers, but the quadratic complexity of attention makes long sequences expensive. State Space Models (SSMs) offer linear complexity, but purely SSM-based architectures flatten the joints into a fixed scan order and capture inter-joint interactions only indirectly.

**STAMPose** (Spatial-Temporal Attention-Mamba Pose estimation) is a hybrid architecture that assigns different operators to the spatial and temporal axes. We select each operator from the accuracy and throughput measured at $J=17$ and $T=243$:

* **Spatial axis ($J=17$)**: attention and a spatial SSM tie in accuracy (40.8 mm), while attention achieves 1.73x higher throughput, so we use attention.
* **Temporal axis ($T=243$)**: the **DC-TSSM (Dual-branch Concatenation Temporal SSM) block**, which fuses a bidirectional selective scan with a depthwise-convolution branch via channel concatenation, is more accurate than temporal attention (40.8 vs. 42.7 mm), so we use DC-TSSM.

On Human3.6M and MPI-INF-3DHP, STAMPose matches or surpasses prior SSM-based models while achieving **1.7–2.0x higher throughput than PoseMamba** at comparable parameter counts.

<!-- **paper** : [ACCV2026](link1) · [arXiv](link2) · [PDF](link3) -->
**paper** : [ACCV2026](link1) · [PDF](link3)

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
git clone https://github.com/akadjsam/STAMPose.git
cd STAMPose

conda create -n STAMPose python=3.8.5
conda activate STAMPose
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cd kernels/selective_scan && pip install -e . && cd ../..
```

## Data Preparation
### Human3.6M
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
@inproceedings{kim2026stampose,
  title     = {STAMPose: Spatial-Temporal Attention-Mamba for Efficient 3D Human Pose Estimation},
  author    = {Kim, Hyun-il and Park, Seung-bo},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## License
This project is released under the [Apache License 2.0](LICENSE).

## Contact
akadjsam@inha.edu or akadjsam@gmail.com
molaal@inha.ac.kr