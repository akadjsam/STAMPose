import argparse
import os
from glob import glob
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import pickle
import sys
import imageio
from tqdm import tqdm
import numpy as np
import os
import cv2
import math
import copy
import imageio
import io
from tqdm import tqdm
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
import os
# 获取当前工作目录
current_directory = os.path.dirname(__file__) + '/../'
sys.path.append(current_directory)
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.tools import ensure_dir
def read_pkl(data_url):
    file = open(data_url, 'rb')
    content = pickle.load(file)
    file.close()
    return content

connections = [
    (10, 9),
    (9, 8),
    (8, 11),
    (8, 14),
    (14, 15),
    (15, 16),
    (11, 12),
    (12, 13),
    (8, 7),
    (7, 0),
    (0, 4),
    (0, 1),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6)
]

def read_h36m(args):
    cam2real = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]], dtype=np.float32)
    scale_factor = 0.5

    sample_joint_seq = read_pkl(current_directory+'data/motion3d/MB3D_f243s81/H36M-SH/test/%08d.pkl' % args.sequence_number)['data_label']
    sample_joint_seq = sample_joint_seq - sample_joint_seq[:,0:1,:]
    sample_joint_seq = sample_joint_seq.transpose(1, 0, 2)
    sample_joint_seq = (sample_joint_seq / scale_factor) @ cam2real
    return sample_joint_seq
def pixel2world_vis_motion(motion, dim=2, is_tensor=False):
#     pose: (17,2,N)
    N = motion.shape[-1]
    if dim==2:
        offset = np.ones([2,N]).astype(np.float32)
    else:
        offset = np.ones([3,N]).astype(np.float32)
        offset[2,:] = 0
    if is_tensor:
        offset = torch.tensor(offset)
    return (motion + offset) * 512 / 2
def get_img_from_fig(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return img
def motion2video_3d(motion, save_path, fps=25, keep_imgs = False):
#     motion: (17,3,N)
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion.shape[-1]
    save_name = save_path.split('.')[0]
    frames = []
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"
    for f in tqdm(range(vlen)):
        j3d = motion[:,:,f]
        fig = plt.figure(0, figsize=(10, 10))
        ax = plt.axes(projection="3d")
        ax.set_xlim(-512, 0)
        ax.set_ylim(-256, 256)
        ax.set_zlim(-512, 0)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        ax.view_init(elev=12., azim=80)
        plt.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)
        for i in range(len(joint_pairs)):
            limb = joint_pairs[i]
            xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
            if joint_pairs[i] in joint_pairs_left:
                ax.plot(-xs, -zs, -ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            elif joint_pairs[i] in joint_pairs_right:
                ax.plot(-xs, -zs, -ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            else:
                ax.plot(-xs, -zs, -ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            
        frame_vis = get_img_from_fig(fig)
        videowriter.append_data(frame_vis)
    videowriter.close()

def main():
    # torch.cuda.set_device(3)
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-number', type=int, default=0)
    parser.add_argument('--dataset', default='h36m')
    parser.add_argument("--config", type=str, default=current_directory+'checkpoint/pose3d/PoseMamba_l/config.yaml', help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default=current_directory+'checkpoint/pose3d/PoseMamba_l', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-e', '--evaluate', default=current_directory+'checkpoint/pose3d/PoseMamba_l/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    configs = get_config(args.config)
    model_backbone = load_backbone(configs)
    # 看情况加DataParallel
    model_backbone = nn.DataParallel(model_backbone)
    # model_backbone = model_backbone.to(device)
    checkpoint = torch.load(args.evaluate, map_location=lambda storage, loc: storage)
    if configs.backbone == 'MotionAGFormer':
        model_backbone.load_state_dict(checkpoint['model'], strict=True)
    else:
        model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model_pos = model_backbone.cuda()
    data_input = read_pkl(current_directory+'data/motion3d/MB3D_f243s81/H36M-SH/test/%08d.pkl' % args.sequence_number)['data_input']
    if configs.no_conf:
        data_input = data_input[ :, :, :2]
    data_label = read_pkl(current_directory+'data/motion3d/MB3D_f243s81/H36M-SH/test/%08d.pkl' % args.sequence_number)['data_label']
    data_input = torch.tensor(data_input).reshape(1,243,17,-1).to(device)
    def flip_data(data):
        """
        horizontal flip
            data: [N, F, 17, D] or [F, 17, D]. X (horizontal coordinate) is the first channel in D.
        Return
            result: same
        """
        left_joints = [4, 5, 6, 11, 12, 13]
        right_joints = [1, 2, 3, 14, 15, 16]
        flipped_data = copy.deepcopy(data)
        flipped_data[..., 0] *= -1                                               # flip x of all joints
        flipped_data[..., left_joints+right_joints, :] = flipped_data[..., right_joints+left_joints, :]
        return flipped_data

    # if configs.flip:    
    #     batch_input_flip = flip_data(data_input)
    #     print(data_input.shape)
    #     predicted_3d_pos_1 = model_pos(data_input)
    #     with torch.no_grad():
    #         predicted_3d_pos_flip = model_pos(batch_input_flip)
    #         predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
    #         predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
    # else:
    #     predicted_3d_pos = model_pos(data_input)


    predicted_3d_pos = model_pos(data_input).squeeze()
    # predicted_3d_pos = predicted_3d_pos - predicted_3d_pos[:,0:1,:]
    predicted_3d_pos = predicted_3d_pos.cpu().detach().numpy()
    # predicted_3d_pos = predicted_3d_pos.squeeze().cpu().detach().numpy()
    print(predicted_3d_pos.shape)
    cam2real = np.array([[1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]], dtype=np.float32)
    scale_factor = 0.5
    predicted_3d_pos = predicted_3d_pos.transpose(1, 0, 2)
    # print(np.mean(np.linalg.norm(predicted_3d_pos - data_label, axis=len(data_label.shape)-1)))
    predicted_3d_pos = (predicted_3d_pos / scale_factor) @ cam2real
    print(f"Visualizing sequence {args.sequence_number} of {args.dataset} dataset")

    def update(frame):
        ax.clear()

        ax.set_xlim3d([min_value[0], max_value[0]])
        ax.set_ylim3d([min_value[1], max_value[1]])
        ax.set_zlim3d([min_value[2], max_value[2]])

        x = sample_joint_seq[:, frame, 0]
        y = sample_joint_seq[:, frame, 1]
        z = sample_joint_seq[:, frame, 2]

        for connection in connections:
            start = sample_joint_seq[connection[0], frame, :]
            end = sample_joint_seq[connection[1], frame, :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]
            ax.plot(xs, ys, zs, c='b')

        ax.scatter(x, y, z)

        return ax,

    dataset_reader_mapper = {
        'h36m': read_h36m,
    }
    sample_joint_seq = dataset_reader_mapper[args.dataset](args)
    print(f'sample_joint_seq shape:{sample_joint_seq.shape}')
    print(f"Number of frames: {sample_joint_seq.shape[1]}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    min_value = np.min(sample_joint_seq, axis=(0, 1))
    max_value = np.max(sample_joint_seq, axis=(0, 1))
    pred_min_value = np.min(predicted_3d_pos, axis=(0, 1))
    pred_max_value = np.max(predicted_3d_pos, axis=(0, 1))
    
    # create the animation
    ani = FuncAnimation(fig, update, frames=sample_joint_seq.shape[1], interval=50)
    # dir_path = 'tools/test'
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)
    dir_path = args.checkpoint + f'/sequence_pose{args.sequence_number}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # ani.save(os.path.join(dir_path, f'{args.dataset}_pose{args.sequence_number}.gif'))
    # 保存每一帧为单独的图片
    for frame in range(predicted_3d_pos.shape[1]):
        ax.clear()

        ax.set_xlim3d([min(min_value[0],pred_min_value[0]), max(max_value[0],pred_max_value[0])])
        ax.set_ylim3d([min(min_value[1],pred_min_value[1]), max(max_value[1],pred_max_value[1])])
        ax.set_zlim3d([min(min_value[2],pred_min_value[2]), max(max_value[2],pred_max_value[2])])

        x = sample_joint_seq[:, frame, 0]
        y = sample_joint_seq[:, frame, 1]
        z = sample_joint_seq[:, frame, 2]

        for connection in connections:
            start = sample_joint_seq[connection[0], frame, :]
            end = sample_joint_seq[connection[1], frame, :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]
            ax.plot(xs, ys, zs, c='0.5')

        ax.scatter(x, y, z)
        plt.savefig(os.path.join(dir_path, f'sequence_{args.sequence_number}_frame_{frame}_gt.png'))
        ax.clear()

        ax.set_xlim3d([min(min_value[0],pred_min_value[0]), max(max_value[0],pred_max_value[0])])
        ax.set_ylim3d([min(min_value[1],pred_min_value[1]), max(max_value[1],pred_max_value[1])])
        ax.set_zlim3d([min(min_value[2],pred_min_value[2]), max(max_value[2],pred_max_value[2])])
        x = predicted_3d_pos[:, frame, 0] 
        y = predicted_3d_pos[:, frame, 1]
        z = predicted_3d_pos[:, frame, 2] 

        for connection in connections:
            start = predicted_3d_pos[connection[0], frame, :]
            end = predicted_3d_pos[connection[1], frame, :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]
            ax.plot(xs, ys, zs, c='b')

        ax.scatter(x, y, z)
        plt.savefig(os.path.join(dir_path, f'sequence_{args.sequence_number}_frame_{frame}_pred.png'))
        ax.clear()

        ax.set_xlim3d([min(min_value[0],pred_min_value[0]), max(max_value[0],pred_max_value[0])])
        ax.set_ylim3d([min(min_value[1],pred_min_value[1]), max(max_value[1],pred_max_value[1])])
        ax.set_zlim3d([min(min_value[2],pred_min_value[2]), max(max_value[2],pred_max_value[2])])
        x = predicted_3d_pos[:, frame, 0] 
        y = predicted_3d_pos[:, frame, 1]
        z = predicted_3d_pos[:, frame, 2] 

        for connection in connections:
            start = predicted_3d_pos[connection[0], frame, :]
            end = predicted_3d_pos[connection[1], frame, :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]
            ax.plot(xs, ys, zs, c='b')

        ax.scatter(x, y, z)
        x = sample_joint_seq[:, frame, 0]
        y = sample_joint_seq[:, frame, 1]
        z = sample_joint_seq[:, frame, 2]

        for connection in connections:
            start = sample_joint_seq[connection[0], frame, :]
            end = sample_joint_seq[connection[1], frame, :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]
            ax.plot(xs, ys, zs, c='0.5')

        ax.scatter(x, y, z)
        plt.savefig(os.path.join(dir_path, f'sequence_{args.sequence_number}_frame_{frame}.png'))



if __name__ == '__main__':
    main()
