import argparse
import os
import pkg_resources
import time
import datetime
import errno

import numpy as np
import scipy.io as scio
import torch
import wandb
import tensorboardX
from torch import optim
from tqdm import tqdm

# Added loss_acceleration
from lib.model.loss import *
# from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity, loss_limb_var, loss_limb_gt, loss_angle, \
#     loss_angle_velocity, loss_acceleration
from lib.utils.data import denormalize
from data.reader.motion_dataset import MPI3DHP, Fusion
from torch.utils.data import DataLoader

from lib.utils.learning import AverageMeter, load_backbone
from lib.utils.tools import count_param_numbers, set_random_seed, get_config
from lib.utils.utils_3dhp import *

# Added logger
from logger import colorlogger

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mpi/TCPFormer_mpi_81.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name")
    parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='checkpoint_mpi', help='new checkpoint directory')
    parser.add_argument('-sd', '--seed', default=1, type=int, help='random seed')
    parser.add_argument('--num-cpus', default=16, type=int, help='Number of CPU cores')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-name', default=None, type=str)
    parser.add_argument('--wandb-run-id', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    opts = parser.parse_args()
    return opts

def get_beijing_timestamp():
    local_offset = time.localtime().tm_gmtoff
    beijing_offset = int(8 * 60 * 60)
    offset = local_offset - beijing_offset
    timestamp = int(datetime.datetime.now().timestamp())
    return timestamp - offset

def train_one_epoch(args, model, train_loader, optimizer, losses):
    model.train()
    for x, y in tqdm(train_loader):
        batch_size = x.shape[0]
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()

        pred = model(x)  # (N, T, 17, 3)

        optimizer.zero_grad()

        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)
        loss_lv = loss_limb_var(pred)
        loss_lg = loss_limb_gt(pred, y)
        loss_a = loss_angle(pred, y)
        loss_av = loss_angle_velocity(pred, y)
        loss_accel = loss_acceleration(pred, y) # added acceleration loss

        loss_total = loss_3d_pos + \
                    args.lambda_scale * loss_3d_scale + \
                    args.lambda_3d_velocity * loss_3d_velocity + \
                    args.lambda_lv * loss_lv + \
                    args.lambda_lg * loss_lg + \
                    args.lambda_a * loss_a + \
                    args.lambda_av * loss_av + \
                    args.Lambda_accel * loss_accel # synced with H36M setting

        losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(), batch_size)
        losses['lg'].update(loss_lg.item(), batch_size)
        losses['angle'].update(loss_a.item(), batch_size)
        losses['angle_velocity'].update(loss_av.item(), batch_size)
        losses['accel'].update(loss_accel.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)

        loss_total.backward()
        optimizer.step()

def input_augmentation(input_2D, model, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape 
    input_2D_flip = input_2D[:, 1]
    input_2D_non_flip = input_2D[:, 0]

    output_3D_flip = model(input_2D_flip)
    output_3D_flip[..., 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D_non_flip = model(input_2D_non_flip)
    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    input_2D = input_2D_non_flip

    return input_2D, output_3D

def evaluate(model, test_loader, n_frames):
    model.eval()
    # Left/Right joint indices for 3DHP (do not modify)
    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]

    data_inference = {}
    error_sum_test = AccumLoss()

    for data in tqdm(test_loader, 0):
        batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_variable('test', [input_2D, gt_3D, batch_cam, scale, bb_box])
        N = input_2D.size(0)

        out_target = gt_3D.clone().view(N, -1, 17, 3)
        out_target[:, :, 14] = 0
        gt_3D = gt_3D.view(N, -1, 17, 3).type(torch.cuda.FloatTensor)

        input_2D, output_3D = input_augmentation(input_2D, model, joints_left, joints_right)

        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1), 17, 3)
        pad = (n_frames - 1) // 2
        pred_out = output_3D[:, pad].unsqueeze(1)

        pred_out[..., 14, :] = 0
        pred_out = denormalize(pred_out, seq)

        pred_out = pred_out - pred_out[..., 14:15, :] 
        inference_out = pred_out + out_target[..., 14:15, :] 
        out_target = out_target - out_target[..., 14:15, :] 

        joint_error_test = mpjpe_cal(pred_out, out_target).item()

        for seq_cnt in range(len(seq)):
            seq_name = seq[seq_cnt]
            if seq_name in data_inference:
                data_inference[seq_name] = np.concatenate(
                    (data_inference[seq_name], inference_out[seq_cnt].permute(2, 1, 0).cpu().numpy()), axis=2)
            else:
                data_inference[seq_name] = inference_out[seq_cnt].permute(2, 1, 0).cpu().numpy()
        
        error_sum_test.update(joint_error_test * N, N)

    for seq_name in data_inference.keys():
        data_inference[seq_name] = data_inference[seq_name][:, :, None, :]
    
    return error_sum_test.avg, data_inference

def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
        'wandb_id': wandb_id,
    }, checkpoint_path)

def save_data_inference(path, data_inference, latest):
    mat_path = os.path.join(path, 'inference_data.mat' if latest else 'inference_data_best.mat')
    scio.savemat(mat_path, data_inference)

def train(args, opts):
    # Timestamp folder creation logic (synced with H36M)
    opts.new_checkpoint = opts.new_checkpoint + '_' + datetime.datetime.fromtimestamp(get_beijing_timestamp()).strftime('%Y_%m_%d_T_%H_%M_%S')
    
    try:
        os.makedirs(opts.new_checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.new_checkpoint)

    global log
    log = colorlogger(opts.new_checkpoint, log_name='log.txt')
    log.info(args)

    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.new_checkpoint, "logs"))

    train_dataset = MPI3DHP(args, train=True)
    test_dataset = Fusion(args, train=False)
    patience_mix = 80
    patience_count = 0
    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': opts.num_cpus - 1 if opts.num_cpus > 1 else 4,
        'pin_memory': True,
        'prefetch_factor': 3,
        'persistent_workers': True
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)
    
    model = load_backbone(args)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[0])
        model = model.cuda()

    n_params = count_param_numbers(model)
    log.info(f"[INFO] Number of parameters: {n_params:,}")

    lr = args.learning_rate
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, amsgrad=True)
    lr_decay = args.lr_decay
    epoch_start = 0
    min_mpjpe = float('inf')  
    wandb_id = opts.wandb_run_id if opts.wandb_run_id is not None else wandb.util.generate_id()

    if opts.checkpoint:
        checkpoint_path = os.path.join(opts.checkpoint, opts.checkpoint_file if opts.checkpoint_file else "latest_epoch.pth.tr")
        if os.path.exists(checkpoint_path):
            log.info(f'Loading checkpoint {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=True)

            if opts.resume:
                lr = checkpoint['lr']
                epoch_start = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                min_mpjpe = checkpoint['min_mpjpe']
                if 'wandb_id' in checkpoint and opts.wandb_run_id is None:
                    wandb_id = checkpoint['wandb_id']
        else:
            log.info("[WARN] Checkpoint path is empty. Starting from the beginning")
            opts.resume = False

    if not opts.eval_only and opts.use_wandb:
        wandb.init(id=wandb_id,
                   name=os.path.basename(opts.new_checkpoint),
                   project='MemoryInducedTransformer',
                   resume="must" if opts.resume else "allow",
                   config=vars(args),
                   settings=wandb.Settings(start_method='fork'))

    checkpoint_path_latest = os.path.join(opts.new_checkpoint, 'latest_epoch.pth.tr')
    checkpoint_path_best = os.path.join(opts.new_checkpoint, 'best_epoch.pth.tr')

    for epoch in range(epoch_start, args.epochs):
        if opts.eval_only:
            with torch.no_grad():
                evaluate(model, test_loader, args.n_frames)
                exit()
            
        log.info(f"Training epoch {epoch}.")
        start_time = time.time()
        
        loss_names = ['3d_pose', '3d_scale', '2d_proj', 'lg', 'lv', '3d_velocity', 'angle', 'angle_velocity', 'accel', 'total']
        losses = {name: AverageMeter() for name in loss_names}
    
        train_one_epoch(args, model, train_loader, optimizer, losses)
        
        with torch.no_grad():
            mpjpe, data_inference = evaluate(model, test_loader, args.n_frames)

        elapsed = (time.time() - start_time) / 60
        log.info('[%d] time %.2f lr %f 3d_train %f mpjpe %f' % (
            epoch + 1, elapsed, lr, losses['3d_pose'].avg, mpjpe))

        # Tensorboard logging
        train_writer.add_scalar('Error MPJPE', mpjpe, epoch + 1)
        train_writer.add_scalar('loss_3d_pos', losses['3d_pose'].avg, epoch + 1)
        train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model, min_mpjpe, wandb_id)
            save_data_inference(opts.new_checkpoint, data_inference, latest=False)
            log.info(f"Best MPJPE updated: {min_mpjpe}")
            patience_count = 0
        else:
            patience_count += 1
            
        save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model, min_mpjpe, wandb_id)
        save_data_inference(opts.new_checkpoint, data_inference, latest=True)

        if opts.use_wandb:
            wandb.log({
                'lr': lr,
                'train/loss_3d_pose': losses['3d_pose'].avg,
                'train/loss_3d_scale': losses['3d_scale'].avg,
                'train/loss_3d_velocity': losses['3d_velocity'].avg,
                'train/loss_accel': losses['accel'].avg,
                'train/total': losses['total'].avg,
                'eval/mpjpe': mpjpe,
                'eval/min_mpjpe': min_mpjpe,
            }, step=epoch + 1)

        # Learning rate decay
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if patience_count > patience_mix:
            log.info('Early stopping triggered.')
            break

    if opts.use_wandb:
        artifact = wandb.Artifact(f'model', type='model')
        artifact.add_file(checkpoint_path_latest)
        artifact.add_file(checkpoint_path_best)
        wandb.log_artifact(artifact)

def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = get_config(opts.config)
    train(args, opts)

if __name__ == '__main__':
    main()