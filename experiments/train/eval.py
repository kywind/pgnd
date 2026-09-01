from pathlib import Path
import random
from tqdm import tqdm, trange

import argparse
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn
import warp as wp
import glob
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import json
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch
from pgnd.material import PGNDModel
from pgnd.data import RealTeleopBatchDataset, RealGripperDataset
from pgnd.utils import Logger, get_root, mkdir

from gs import do_gs
from pv_train import do_train_pv
from pv_dataset import do_dataset_pv
from metric_eval import do_metric
from train_eval import transform_gripper_points, dataloader_wrapper

root: Path = get_root(__file__)


def eval(
    cfg: DictConfig,
    ckpt_path: str,
    episode: int,
    dataset_pv: bool = True,
    eval_base_name: str = 'eval',
    use_pv: bool = True,
    use_gs: bool = True,
):

    # init
    wp.init()
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})

    gpus = [int(gpu) for gpu in cfg.gpus]
    wp_devices = [wp.get_device(f'cuda:{gpu}') for gpu in gpus]
    torch_devices = [torch.device(f'cuda:{gpu}') for gpu in gpus]
    device_count = len(torch_devices)
    
    assert device_count == 1
    wp_device = wp_devices[0]
    torch_device = torch_devices[0]

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.autograd.set_detect_anomaly(True)

    torch.backends.cudnn.benchmark = True

    log_root: Path = root / 'log'
    eval_name = f'{cfg.train.name}/{eval_base_name}/{cfg.train.dataset_name.split("/")[-1]}/{cfg.iteration:06d}'
    exp_root: Path = log_root / eval_name
    state_root: Path = exp_root / 'state'
    mkdir(state_root, overwrite=cfg.overwrite, resume=cfg.resume)
    episode_state_root = state_root / f'episode_{episode:04d}'
    mkdir(episode_state_root, overwrite=cfg.overwrite, resume=cfg.resume)
    OmegaConf.save(cfg, exp_root / 'hydra.yaml', resolve=True)

    use_pv = cfg.train.use_pv
    if not use_pv:
        print('not using pv rendering...')
    
    # decide whether to use gs rendering based on the existence of gs files
    assert os.path.exists(log_root / str(cfg.train.source_dataset_name) / f'episode_{episode:04d}' / 'meta.txt')
    meta = np.loadtxt(log_root / str(cfg.train.source_dataset_name) / f'episode_{episode:04d}' / 'meta.txt')
    with open(log_root / str(cfg.train.source_dataset_name) / 'metadata.json') as f:
        datadir_list = json.load(f)
    datadir = datadir_list[episode]
    source_data_dir = datadir['path']
    source_episode_id = int(meta[0])
    source_frame_start = int(meta[1]) + int(cfg.sim.n_history) * int(cfg.train.dataset_load_skip_frame) * int(cfg.train.dataset_skip_frame)
    source_frame_end = int(meta[2])
    if use_gs:
        use_gs = os.path.exists((log_root.parent.parent / source_data_dir).parent / f'episode_{source_episode_id:04d}' / 'gs' / f'{source_frame_start:06d}.splat')

    if cfg.train.dataset_name is None:
        cfg.train.dataset_name = Path(cfg.train.name).parent / 'dataset'
    assert cfg.train.source_dataset_name is not None

    source_dataset_root = log_root / str(cfg.train.source_dataset_name)
    assert os.path.exists(source_dataset_root)

    dataset = RealTeleopBatchDataset(
        cfg, 
        dataset_root=log_root / cfg.train.dataset_name / 'state',
        source_data_root=source_dataset_root,
        device=torch_device,
        num_steps=cfg.sim.num_steps,
        eval_episode_name=f'episode_{episode:04d}',
    )
    dataloader = dataloader_wrapper(
        DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True),
        'dataset'
    )
    if cfg.sim.gripper_points:
        eval_gripper_dataset = RealGripperDataset(
            cfg,
            device=torch_device,
        )
        eval_gripper_dataloader = dataloader_wrapper(
            DataLoader(eval_gripper_dataset, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True),
            'gripper_dataset'
        )

    # load ckpt
    if ckpt_path is None:
        if cfg.model.ckpt is not None:
            ckpt_path = cfg.model.ckpt
        else:
            ckpt_path = log_root / cfg.train.name / 'ckpt' / f'{cfg.iteration:06d}.pt'
    ckpt = torch.load(log_root / ckpt_path, map_location=torch_device)
    
    material: nn.Module = PGNDModel(cfg)
    material.to(torch_device)
    material.load_state_dict(ckpt['material'])
    material.requires_grad_(False)
    material.eval()

    if 'friction' in ckpt:
        friction = ckpt['friction']['mu'].reshape(-1, 1)
    else:
        friction = torch.tensor(cfg.model.friction.value, device=torch_device).reshape(-1, 1)

    init_state, actions, gt_states, downsample_indices = next(dataloader)

    x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
    x = x.to(torch_device)
    v = v.to(torch_device)
    x_his = x_his.to(torch_device)
    v_his = v_his.to(torch_device)

    actions = actions.to(torch_device)

    if cfg.sim.gripper_points:
        gripper_points, _ = next(eval_gripper_dataloader)
        gripper_points = gripper_points.to(torch_device)
        gripper_x, gripper_v, gripper_mask = transform_gripper_points(cfg, gripper_points, actions)  # (bsz, num_steps, num_grippers, 3)

    gt_x, gt_v = gt_states
    gt_x = gt_x.to(torch_device)
    gt_v = gt_v.to(torch_device)

    # gt_states: (bsz, num_steps_total)
    batch_size = gt_x.shape[0]
    num_steps_total = gt_x.shape[1]
    num_particles = gt_x.shape[2]
    assert batch_size == 1

    if cfg.sim.gripper_points:
        num_gripper_particles = gripper_x.shape[2]
        num_particles_orig = num_particles
        num_particles = num_particles + num_gripper_particles

    cfg.sim.num_steps = num_steps_total
    sim = CacheDiffSimWithFrictionBatch(cfg, num_steps_total, batch_size, wp_device, requires_grad=True)

    statics = StaticsBatch()
    statics.init(shape=(batch_size, num_particles), device=wp_device)
    statics.update_clip_bound(clip_bound)
    statics.update_enabled(enabled)
    colliders = CollidersBatch()

    if cfg.sim.gripper_points:
        assert not cfg.sim.gripper_forcing
        num_grippers = 0
    else:
        num_grippers = cfg.sim.num_grippers

    colliders.init(shape=(batch_size, num_grippers), device=wp_device)
    if num_grippers > 0:
        assert len(actions.shape) > 2
        colliders.initialize_grippers(actions[:, 0])

    ckpt_init = dict(x=x[0], v=v[0], grippers=actions[0, 0].clone())
    if cfg.sim.gripper_points:
        ckpt_init['gripper_x'] = gripper_x[0, 0]
        ckpt_init['gripper_v'] = gripper_v[0, 0]
    torch.save(ckpt_init, episode_state_root / f'{0:04d}.pt')

    enabled = enabled.to(torch_device)  # (bsz, num_particles)
    enabled_mask = enabled.unsqueeze(-1).repeat(1, 1, 3)  # (bsz, num_particles, 3)

    losses = {}
    with torch.no_grad():
        for step in trange(num_steps_total):
            if num_grippers > 0:
                colliders.update_grippers(actions[:, step])
            x_in = x.clone()
            v_in = v.clone()

            if cfg.sim.gripper_points:
                x = torch.cat([x, gripper_x[:, step]], dim=1)  # gripper_points: (bsz, num_steps, num_particles, 3)
                v = torch.cat([v, gripper_v[:, step]], dim=1)
                x_his = torch.cat([x_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=x_his.device, dtype=x_his.dtype)], dim=1)
                v_his = torch.cat([v_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=v_his.device, dtype=v_his.dtype)], dim=1)
                if enabled.shape[1] < num_particles:
                    enabled = torch.cat([enabled, gripper_mask[:, step]], dim=1)
                statics.update_enabled(enabled.cpu())

            pred = material(x, v, x_his, v_his, enabled)

            if pred.isnan().any():
                print('pred isnan', pred.min().item(), pred.max().item())
                break
            if pred.isinf().any():
                print('pred isinf', pred.min().item(), pred.max().item())
                break

            x, v = sim(statics, colliders, step, x, v, friction, pred)

            if cfg.sim.gripper_forcing:
                assert not cfg.sim.gripper_points
                gripper_xyz = actions[:, step, :, :3]
                gripper_v = actions[:, step, :, 3:6]
                x_from_gripper = x_in[:, None] - gripper_xyz[:, :, None]  # (bsz, num_grippers, num_particles, 3)
                x_gripper_distance = torch.norm(x_from_gripper, dim=-1)  # (bsz, num_grippers, num_particles)
                x_gripper_distance_mask = x_gripper_distance < cfg.model.gripper_radius
                x_gripper_distance_mask = x_gripper_distance_mask.unsqueeze(-1).repeat(1, 1, 1, 3)  # (bsz, num_grippers, num_particles, 3)
                gripper_v_expand = gripper_v[:, :, None].repeat(1, 1, num_particles, 1)  # (bsz, num_grippers, num_particles, 3)

                gripper_closed = actions[:, step, :, -1] < 0.5  # (bsz, num_grippers)  # 1: open, 0: close
                x_gripper_distance_mask = torch.logical_and(x_gripper_distance_mask, gripper_closed[:, :, None, None].repeat(1, 1, num_particles, 3))

                gripper_quat_vel = actions[:, step, :, 10:13]  # (bsz, num_grippers, 3)
                gripper_angular_vel = torch.linalg.norm(gripper_quat_vel, dim=-1, keepdims=True)  # (bsz, num_grippers, 1)
                gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)  # (bsz, num_grippers, 3)

                grid_from_gripper_axis = x_from_gripper - \
                    (gripper_quat_axis[:, :, None] * x_from_gripper).sum(dim=-1, keepdims=True) * gripper_quat_axis[:, :, None]  # (bsz, num_grippers, num_particles, 3)
                gripper_v_expand = torch.cross(gripper_quat_vel[:, :, None], grid_from_gripper_axis, dim=-1) + gripper_v_expand

                for i in range(gripper_xyz.shape[1]):
                    x_gripper_distance_mask_single = x_gripper_distance_mask[:, i]
                    x[x_gripper_distance_mask_single] = x_in[x_gripper_distance_mask_single] + cfg.sim.dt * gripper_v_expand[:, i][x_gripper_distance_mask_single]
                    v[x_gripper_distance_mask_single] = gripper_v_expand[:, i][x_gripper_distance_mask_single]
            
            if cfg.sim.n_history > 0:
                if cfg.sim.gripper_points:
                    x_his_particles = torch.cat([x_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], x_in[:, :num_particles_orig, None].detach()], dim=2)
                    v_his_particles = torch.cat([v_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], v_in[:, :num_particles_orig, None].detach()], dim=2)
                    x_his = x_his_particles.reshape(batch_size, num_particles_orig, -1)
                    v_his = v_his_particles.reshape(batch_size, num_particles_orig, -1)
                else:
                    x_his = torch.cat([x_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], x_in[:, :, None].detach()], dim=2)
                    v_his = torch.cat([v_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], v_in[:, :, None].detach()], dim=2)
                    x_his = x_his.reshape(batch_size, num_particles, -1)
                    v_his = v_his.reshape(batch_size, num_particles, -1)

            if cfg.sim.gripper_points:
                extra_save = {
                    'gripper_x': gripper_x[0, step],
                    'gripper_v': gripper_v[0, step],
                }
                x = x[:, :num_particles_orig]
                v = v[:, :num_particles_orig]
                enabled = enabled[:, :num_particles_orig]
            else:
                extra_save = {}

            grippers_save = actions[0, step].clone()
            grippers_save[:, :3] = grippers_save[:, :3] + grippers_save[:, 3:6] * cfg.sim.dt * cfg.sim.interval

            loss_x = nn.functional.mse_loss(x[enabled_mask > 0], gt_x[:, step][enabled_mask > 0])
            loss_v = nn.functional.mse_loss(v[enabled_mask > 0], gt_v[:, step][enabled_mask > 0])
            losses[step] = dict(loss_x=loss_x.item(), loss_v=loss_v.item())

            ckpt = dict(x=x[0], v=v[0], grippers=grippers_save, **extra_save)

            if (step + 1) % cfg.sim.skip_frame == 0:
                torch.save(ckpt, episode_state_root / f'{int((step + 1) / cfg.sim.skip_frame):04d}.pt')

    for loss_k in losses[0].keys():
        plt.figure(figsize=(10, 5))
        loss_list = [losses[step][loss_k] for step in losses]
        plt.plot(loss_list)
        plt.title(loss_k)
        plt.grid()
        plt.savefig(state_root / f'episode_{episode:04d}_{loss_k}.png', dpi=300)

    ## pv
    if use_pv:
        do_train_pv(
            cfg,
            log_root,
            cfg.iteration,
            [f'episode_{episode:04d}'],
            eval_dirname=eval_base_name,
            dataset_name=cfg.train.dataset_name.split("/")[-1],
            eval_postfix='',
        )

    if use_gs:
        do_gs(
            cfg,
            log_root,
            cfg.iteration,
            [f'episode_{episode:04d}'],
            eval_dirname=eval_base_name,
            dataset_name=cfg.train.dataset_name.split("/")[-1],
            eval_postfix='',
            camera_id=1,
            with_mask=True,
            with_bg=True,
        )

    if use_pv:
        save_dir = log_root / f'{cfg.train.name}/{eval_base_name}/{cfg.train.dataset_name.split("/")[-1]}/{cfg.iteration:06d}/pv'
        _ = do_dataset_pv(
            cfg,
            log_root / str(cfg.train.dataset_name),
            [f'episode_{episode:04d}'],
            save_dir=save_dir,
            downsample_indices=downsample_indices,
        )
    
    metrics = do_metric(
        cfg,
        log_root,
        cfg.iteration,
        [f'episode_{episode:04d}'],
        downsample_indices,
        eval_dirname=eval_base_name,
        dataset_name=cfg.train.dataset_name.split("/")[-1],
        eval_postfix='',
        camera_id=1,
        use_gs=use_gs,
    )
    return metrics


@torch.no_grad()
def main(
    cfg: DictConfig, 
):

    print(OmegaConf.to_yaml(cfg, resolve=True))

    metrics_list = []
    for episode in range(cfg.start_episode, cfg.end_episode):
        if "eval_state_only" in cfg and cfg.eval_state_only:
            use_pv = False
            use_gs = False
        else:
            use_pv = True
            use_gs = True
        metrics = eval(cfg,
            None,
            episode,
            dataset_pv=True,
            eval_base_name='eval',
            use_pv=use_pv, 
            use_gs=use_gs,
        )
        metrics_list.append(metrics)

    metrics_list = np.array(metrics_list)[:, 0]

    if metrics_list.shape[-1] == 10:
        metric_names = ['mde', 'chamfer', 'emd', 'jscore', 'fscore', 'jfscore', 'perception', 'psnr', 'ssim', 'iou']
    else:
        assert metrics_list.shape[-1] == 3
        metric_names = ['mde', 'chamfer', 'emd']

    median_metric = np.median(metrics_list, axis=0)
    step_75_metric = np.percentile(metrics_list, 75, axis=0)
    step_25_metric = np.percentile(metrics_list, 25, axis=0)

    for i, metric_name in enumerate(metric_names):
        # plot error
        x = np.arange(1, len(median_metric) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(x, median_metric[:, i])
        plt.xlabel(f"prediction steps, dt={cfg.sim.dt}")
        plt.ylabel(metric_name)
        plt.grid()

        ax = plt.gca()
        x = np.arange(1, len(median_metric) + 1)
        ax.fill_between(x, step_25_metric[:, i], step_75_metric[:, i], alpha=0.2)

        save_dir = root / 'log' / cfg.train.name / eval_base_name / cfg.train.dataset_name.split("/")[-1] / f'{cfg.iteration:06d}' / 'metric'
        plt.savefig(os.path.join(save_dir, f'{i:02d}-{metric_name}.png'))
        plt.close()

    mean_metric = np.mean(metrics_list, axis=0)
    std_metric = np.std(metrics_list, axis=0)

    n_steps = 30
    mean_metric_step = mean_metric[n_steps]
    std_metric_step = std_metric[n_steps]

    if mean_metric.shape[-1] == 10:
        mde, chamfer, emd, jscore, fscore, jfscore, perception, psnr, ssim, iou = mean_metric_step
        mde_std, chamfer_std, emd_std, jscore_std, fscore_std, jfscore_std, perception_std, psnr_std, ssim_std, iou_std = std_metric_step
        print(f'3D MDE: {mde:.4f} {mde_std:.4f}, 3D CD: {chamfer:.4f} {chamfer_std:.4f}, 3D EMD: {emd:.4f} {emd_std:.4f}', end=' ')
        print(f'J-Score: {jscore:.4f} {jscore_std:.4f}, F-Score: {fscore:.4f} {fscore_std:.4f}, JF-Score: {jfscore:.4f} {jfscore_std:.4f}', end=' ')
        print(f'perception: {perception:.4f} {perception_std:.4f}, PSNR: {psnr:.4f} {psnr_std:.4f}, SSIM: {ssim:.4f} {ssim_std:.4f}, IoU: {iou:.4f} {iou_std:.4f}')
    else:
        mde, chamfer, emd = mean_metric_step
        mde_std, chamfer_std, emd_std = std_metric_step
        print(f'3D MDE: {mde:.4f} {mde_std:.4f}, 3D CD: {chamfer:.4f} {chamfer_std:.4f}, 3D EMD: {emd:.4f} {emd_std:.4f}')


if __name__ == '__main__':

    best_models = {
        'cloth': ['cloth', 'train', 100000],
        'rope': ['rope', 'train', 100000],
        'paperbag': ['paperbag', 'train', 100000],
        'sloth': ['sloth', 'train', 100000],
        'box': ['box', 'train', 100000],
        'bread': ['bread', 'train', 100000],
    }

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--task', type=str, required=True)
    arg_parser.add_argument('--state_only', action='store_true')
    args = arg_parser.parse_args()

    with open(root / f'log/{best_models[args.task][0]}/{best_models[args.task][1]}/hydra.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    cfg = OmegaConf.create(config)

    cfg.iteration = best_models[args.task][2]
    cfg.start_episode = cfg.train.eval_start_episode
    cfg.end_episode = cfg.train.eval_end_episode
    cfg.sim.num_steps = 1000
    cfg.sim.gripper_forcing = False
    cfg.sim.uniform = True
    cfg.sim.use_pv = True
    cfg.eval_state_only = args.state_only

    main(cfg)
