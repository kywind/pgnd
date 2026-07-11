from pathlib import Path
import random
import time
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
import os
import glob
from PIL import Image
import argparse
import yaml
import scipy
import math
import lpips
import cv2
from skimage.metrics import structural_similarity as ssim_sk
import json

from pgnd.utils import get_root
from pgnd.ffmpeg import make_video
from train.metric import mean_dist, chamfer_dist, em_distance, compute_j, compute_f, compute_lpips, calc_psnr, calc_ssim, inverse_preprocess

root: Path = get_root(__file__)

@torch.no_grad()
def do_metric(
    cfg,
    log_root,
    iteration,
    episode_names,
    downsample_indices,
    eval_dirname,
    camera_id=1,
    eval_postfix='',
    dataset_name='',
    use_gs=True,
    eval_camera_num=0,
):

    state_dir = log_root / cfg.train.name / eval_dirname / dataset_name / f'{iteration:06d}' / 'state'

    if use_gs:
        pv_gs_dir = log_root / cfg.train.name / eval_dirname / dataset_name / f'{iteration:06d}' / 'pv_gs'
        mask_dir = log_root / cfg.train.name / eval_dirname / dataset_name / f'{iteration:06d}' / 'mask'
        
        pv_gs_gt_dir = log_root / cfg.train.name / eval_dirname / dataset_name / f'{iteration:06d}' / 'pv_gs_gt'
        mask_gt_dir = log_root / cfg.train.name / eval_dirname / dataset_name / f'{iteration:06d}' / 'mask_gt'

    save_dir = log_root / cfg.train.name / eval_dirname / dataset_name / f'{iteration:06d}' / 'metric'
    save_dir.mkdir(parents=True, exist_ok=True)

    loss_fn_vgg = lpips.LPIPS(net='alex')

    metric_list_list = []

    for episode_idx, episode in enumerate(episode_names):
        state_dir_episode = state_dir / episode
        save_dir_episode = save_dir / episode
        save_dir_episode.mkdir(parents=True, exist_ok=True)

        if use_gs:
            pv_gs_dir_episode = pv_gs_dir / episode
            mask_dir_episode = mask_dir / episode
            pv_gs_gt_dir_episode = pv_gs_gt_dir / episode
            mask_gt_dir_episode = mask_gt_dir / episode
            pv_gs_gt_dir_episode.mkdir(parents=True, exist_ok=True)
            mask_gt_dir_episode.mkdir(parents=True, exist_ok=True)

        # save downsample_indices
        downsample_indices = downsample_indices[0].cuda()
        np.save(save_dir_episode / 'downsample_indices.npy', downsample_indices.cpu().numpy())

        source_dataset_root = log_root / str(cfg.train.source_dataset_name)

        meta = np.loadtxt(log_root / str(cfg.train.source_dataset_name) / episode / 'meta.txt')
        with open(log_root / str(cfg.train.source_dataset_name) / 'metadata.json') as f:
            datadir_list = json.load(f)
        episode_real_name = int(episode.split('_')[1])
        datadir = datadir_list[episode_real_name]
        source_data_dir = datadir['path']
        source_episode_id = int(meta[0])
        source_frame_start = int(meta[1])
        source_frame_end = int(meta[2])

        skip_frame = cfg.train.dataset_load_skip_frame * cfg.train.dataset_skip_frame
        frame_ids = np.arange(source_frame_start + cfg.sim.n_history * skip_frame, source_frame_end, cfg.sim.skip_frame * skip_frame)
        n_frames = len(frame_ids)

        # load xyz_orig for inverse preprocess
        xyz_orig = np.load(source_dataset_root / episode / 'traj.npz')['xyz']
        xyz_orig = torch.tensor(xyz_orig, dtype=torch.float32)

        traj = []
        if use_gs:
            imgs = []
            masks = []
            gt_imgs = []
            gt_masks = []

        for frame_id in range(n_frames):
            frame_id_gt = frame_ids[frame_id]

            state = torch.load(state_dir_episode / f'{frame_id:04d}.pt')
            x = state['x'].cpu()
            traj.append(x)  # (n, 3)

            if use_gs:
                pv_gs = cv2.imread(pv_gs_dir_episode / f'{frame_id:04d}.png')
                pv_gs = cv2.cvtColor(pv_gs, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_dir_episode / f'{frame_id:04d}.png')
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                imgs.append(pv_gs)  # (H, W, 3)
                masks.append(mask)  # (H, W, 3)

                gt_mask = cv2.imread((log_root.parent.parent / source_data_dir).parent / f'episode_{source_episode_id:04d}' \
                        / f'camera_{camera_id}' / 'mask' / f'{int(frame_id_gt):06d}.png')
                gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
                gt_masks.append(gt_mask)

                gt_img = cv2.imread((log_root.parent.parent / source_data_dir).parent / f'episode_{source_episode_id:04d}' \
                        / f'camera_{camera_id}' / 'rgb' / f'{int(frame_id_gt):06d}.jpg')
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                gt_img = gt_img * (gt_mask > 0)[..., None]
                gt_imgs.append(gt_img)

                # save
                pv_gs_gt = Image.fromarray(gt_img)
                pv_gs_gt.save(pv_gs_gt_dir_episode / f'{frame_id:04d}.png')
                mask_gt = Image.fromarray(gt_mask)
                mask_gt.save(mask_gt_dir_episode / f'{frame_id:04d}.png')
        
        if use_gs:
            frame_rate = 10
            video_name = pv_gs_gt_dir / f'{episode}.mp4'
            make_video(pv_gs_gt_dir_episode, video_name, '%04d.png', frame_rate)
        
        traj = torch.stack(traj, dim=0)
        traj, xyz_orig = inverse_preprocess(cfg, traj, xyz_orig)
        gt_traj = xyz_orig[cfg.sim.n_history * skip_frame::cfg.sim.skip_frame * skip_frame]

        if use_gs:
            assert len(imgs) == len(gt_imgs) == len(gt_masks) == len(traj) == len(gt_traj)
        else:
            assert len(traj) == len(gt_traj)

        metric_list = []
        for i in range(1, len(traj)):
            xyz = traj[i].cuda()
            xyz_gt = gt_traj[i].cuda()

            if use_gs:
                im = imgs[i]
                mask = masks[i]
                im_gt = gt_imgs[i]
                mask_gt = gt_masks[i]

                mask = mask > 0
                mask_gt = mask_gt > 0

            xyz_gt_downsampled = xyz_gt[downsample_indices]

            mde = mean_dist(xyz, xyz_gt_downsampled)
            chamfer = chamfer_dist(xyz, xyz_gt)
            emd = em_distance(xyz, xyz_gt)

            if use_gs:
                jscore = compute_j(mask, mask_gt)
                fscore = compute_f(mask, mask_gt)
                jfscore = (jscore + fscore) / 2

                perception = compute_lpips(loss_fn_vgg, im, im_gt)
                psnr = calc_psnr(im, im_gt, mask_gt)
                ssim = calc_ssim(im, im_gt, mask_gt)
                iou = np.sum(mask & mask_gt) / np.sum(mask | mask_gt)

                metric_list.append([mde, chamfer, emd, jscore, fscore, jfscore, perception, psnr, ssim, iou])
                print(f'{episode}, image: {i}, camera: {camera_id}', end=' ')
                print(f'3D MDE: {mde:.4f}, 3D CD: {chamfer:.4f}, 3D EMD: {emd:.4f}', end=' ')
                print(f'J-Score: {jscore:.4f}, F-Score: {fscore:.4f}, JF-Score: {jfscore:.4f}', end=' ')
                print(f'perception: {perception:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, IoU: {iou:.4f}')

            else:
                metric_list.append([mde, chamfer, emd])
                print(f'{episode}, image: {i}', end=' ')
                print(f'3D MDE: {mde:.4f}, 3D CD: {chamfer:.4f}, 3D EMD: {emd:.4f}')

        # save metrics
        metric_list = np.array(metric_list)
        np.savetxt(save_dir_episode / f'metric.txt', metric_list, fmt='%.6f')

        metric_list_list.append(metric_list)

    metric_list_list = np.array(metric_list_list)
    return metric_list_list
