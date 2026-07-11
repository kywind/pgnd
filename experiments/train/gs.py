from pathlib import Path
import random
from tqdm import tqdm, trange
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn
import math
import os
import cv2
from sklearn.neighbors import NearestNeighbors
import json
import kornia

from pgnd.utils import get_root, mkdir
from pgnd.ffmpeg import make_video

from real_world.utils.render_utils import interpolate_motions
from real_world.gs.helpers import setup_camera
from real_world.gs.convert import save_to_splat, read_splat

from diff_gaussian_rasterization import GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

root: Path = get_root(__file__)


def Rt_to_w2c(R, t):
    c2w = np.concatenate([np.concatenate([R, t.reshape(3, 1)], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
    w2c = np.linalg.inv(c2w)
    return w2c


class GSRenderer:
    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        self.device = device
        self.k_rel = 16  # knn for relations
        self.k_wgt = 16  # knn for weights
        self.clear()
    
    def clear(self, clear_params=True):
        self.metadata = None
        self.config = None
        if clear_params:
            self.params = None
    
    def load_params(self, params_path, remove_low_opa=True, remove_black=False):
        pts, colors, scales, quats, opacities = read_splat(params_path)
        
        if remove_low_opa:
            low_opa_idx = opacities[:, 0] < 0.1
            pts = pts[~low_opa_idx]
            colors = colors[~low_opa_idx]
            quats = quats[~low_opa_idx]
            opacities = opacities[~low_opa_idx]
            scales = scales[~low_opa_idx]

        if remove_black:
            low_color_idx = colors.sum(axis=-1) < 0.5
            pts = pts[~low_color_idx]
            colors = colors[~low_color_idx]
            quats = quats[~low_color_idx]
            opacities = opacities[~low_color_idx]
            scales = scales[~low_color_idx]
        
        self.params = {
            'means3D': torch.from_numpy(pts).to(self.device),
            'rgb_colors': torch.from_numpy(colors).to(self.device),
            'log_scales': torch.log(torch.from_numpy(scales).to(self.device)),
            'unnorm_rotations': torch.from_numpy(quats).to(self.device),
            'logit_opacities': torch.logit(torch.from_numpy(opacities).to(self.device))
        }

        gripper_splat = root / 'log/gs/ckpts/gripper.splat'  # gripper_new.splat
        table_splat = root / 'log/gs/ckpts/table.splat'

        self.gripper_params = read_splat(gripper_splat)
        self.table_params = read_splat(table_splat)

    def set_camera(self, w, h, intr, w2c=None, R=None, t=None, near=0.01, far=100.0):
        if w2c is None:
            assert R is not None and t is not None
            w2c = Rt_to_w2c(R, t)
        self.metadata = {
            'w': w,
            'h': h,
            'k': intr,
            'w2c': w2c,
        }
        self.config = {'near': near, 'far': far}

    @torch.no_grad
    def render(self, render_data, cam_id, bg=[0, 0, 0]):
        render_data = {k: v.to(self.device) for k, v in render_data.items()}
        w, h = self.metadata['w'], self.metadata['h']
        k, w2c = self.metadata['k'], self.metadata['w2c']
        cam = setup_camera(w, h, k, w2c, self.config['near'], self.config['far'], bg)
        im, _, depth, = GaussianRasterizer(raster_settings=cam)(**render_data)
        return im, depth
    
    def knn_relations(self, bones):
        k = self.k_rel
        knn = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(bones.detach().cpu().numpy())
        _, indices = knn.kneighbors(bones.detach().cpu().numpy())  # (N, k)
        indices = indices[:, 1:]  # exclude self
        return indices
    
    def knn_weights(self, bones, pts):
        k = self.k_wgt
        knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(bones.detach().cpu().numpy())
        _, indices = knn.kneighbors(pts.detach().cpu().numpy())
        bones_selected = bones[indices]  # (N, k, 3)
        dist = torch.norm(bones_selected - pts[:, None], dim=-1)  # (N, k)
        weights = 1 / (dist + 1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # (N, k)
        weights_all = torch.zeros((pts.shape[0], bones.shape[0]), device=pts.device)
        weights_all[torch.arange(pts.shape[0])[:, None], indices] = weights
        return weights_all

    def rollout_and_render(self, pts_list, grippers=[], with_bg=False):
        assert self.params is not None

        pts_list = pts_list.to(self.device)

        if grippers != []:
            n_grippers = grippers.shape[1]
            grippers = grippers.to(self.device)
            gripper_center = grippers[:, :, :3]
            gripper_quat = grippers[:, :, 6:10]
            gripper_radius = grippers[:, :, 13]

        xyz_0 = self.params['means3D']
        rgb_0 = self.params['rgb_colors']
        quat_0 = torch.nn.functional.normalize(self.params['unnorm_rotations'])
        opa_0 = torch.sigmoid(self.params['logit_opacities'])
        scales_0 = torch.exp(self.params['log_scales'])

        pts_prev = pts_list[0]

        xyz_list = [xyz_0]
        rgb_list = [rgb_0]
        quat_list = [quat_0]
        opa_list = [opa_0]
        scales_list = [scales_0]
        for i in range(1, len(pts_list)):
            pts = pts_list[i]

            xyz, quat, _ = interpolate_motions(
                bones=pts_prev,
                motions=pts - pts_prev,
                relations=self.knn_relations(pts_prev),
                weights=self.knn_weights(pts_prev, xyz_list[-1]),
                xyz=xyz_list[-1],
                quat=quat_list[-1],
                step=f'{i-1}->{i}'
            )

            pts_prev = pts
            xyz_list.append(xyz)
            quat_list.append(quat)
            rgb_list.append(rgb_list[-1])
            opa_list.append(opa_list[-1])
            scales_list.append(scales_list[-1])
        
        n_steps = len(xyz_list)
        xyz = torch.stack(xyz_list, dim=0).to(torch.float32)
        rgb = torch.stack(rgb_list, dim=0).to(torch.float32)
        quat = torch.stack(quat_list, dim=0).to(torch.float32)
        opa = torch.stack(opa_list, dim=0).to(torch.float32)
        scales = torch.stack(scales_list, dim=0).to(torch.float32)

        # interpolate smoothly
        change_points = (xyz - torch.concatenate([xyz[0:1], xyz[:-1]], dim=0)).norm(dim=-1).sum(dim=-1).nonzero().squeeze(1)
        change_points = torch.cat([torch.tensor([0]).to(change_points.device), change_points])
        for i in range(1, len(change_points)):
            start = change_points[i - 1]
            end = change_points[i]
            if end - start < 2:  # gap is 0 or 1
                continue
            xyz[start:end] = torch.lerp(xyz[start][None], xyz[end][None], torch.linspace(0, 1, end - start + 1).to(xyz.device)[:, None, None])[:-1]
            rgb[start:end] = torch.lerp(rgb[start][None], rgb[end][None], torch.linspace(0, 1, end - start + 1).to(rgb.device)[:, None, None])[:-1]
            quat[start:end] = torch.lerp(quat[start][None], quat[end][None], torch.linspace(0, 1, end - start + 1).to(quat.device)[:, None, None])[:-1]
            opa[start:end] = torch.lerp(opa[start][None], opa[end][None], torch.linspace(0, 1, end - start + 1).to(opa.device)[:, None, None])[:-1]

        quat = torch.nn.functional.normalize(quat, dim=-1)
        mean_xyz = xyz.mean((0, 1))

        if with_bg:
            ## add table and gripper
            # add table
            t_pts, t_colors, t_scales, t_quats, t_opacities = self.table_params
            t_pts = torch.tensor(t_pts).to(xyz.device).to(xyz.dtype)
            t_colors = torch.tensor(t_colors).to(rgb.device).to(rgb.dtype)
            t_scales = torch.tensor(t_scales).to(scales.device).to(scales.dtype)
            t_quats = torch.tensor(t_quats).to(quat.device).to(quat.dtype)
            t_opacities = torch.tensor(t_opacities).to(opa.device).to(opa.dtype)

            # add table pos
            t_pts = t_pts + torch.tensor([mean_xyz[0].item() - 0.36, mean_xyz[1].item() - 0.10, 0.02]).to(t_pts.device).to(t_pts.dtype)

            # add gripper
            g_pts, g_colors, g_scales, g_quats, g_opacities = self.gripper_params
            g_pts = torch.tensor(g_pts).to(xyz.device).to(xyz.dtype)
            g_colors = torch.tensor(g_colors).to(rgb.device).to(rgb.dtype)
            g_scales = torch.tensor(g_scales).to(scales.device).to(scales.dtype)
            g_quats = torch.tensor(g_quats).to(quat.device).to(quat.dtype)
            g_opacities = torch.tensor(g_opacities).to(opa.device).to(opa.dtype)

            g_pts_tip = g_pts[(g_pts[:, 2] > -0.10) & (g_pts[:, 2] < -0.02)]
            g_pts_tip_mean_xy = g_pts_tip[:, :2].mean(dim=0)
            g_pts_translation = torch.tensor([-g_pts_tip_mean_xy[0] - 0.02, -g_pts_tip_mean_xy[1] + 0.0, 0.07]).to(g_pts.device).to(g_pts.dtype)
            g_pts = g_pts + g_pts_translation

            # rotate gripper
            gripper_mat = kornia.geometry.conversions.quaternion_to_rotation_matrix(gripper_quat)  # (num_steps, num_grippers, 3, 3)
            g_pts = g_pts @ gripper_mat  # (num_steps, num_grippers, num_points, 3)

            g_quats_mat = kornia.geometry.conversions.quaternion_to_rotation_matrix(g_quats)  # (num_grippers, 3, 3)
            g_quats_mat = g_quats_mat[None, None].repeat(n_steps, n_grippers, 1, 1, 1)  # (num_steps, num_grippers, num_points, 3, 3)
            g_quats_mat = gripper_mat.permute(0, 1, 3, 2)[:, :, None] @ g_quats_mat  # (num_steps, num_grippers, num_points, 3, 3)
            g_quats = kornia.geometry.conversions.rotation_matrix_to_quaternion(g_quats_mat)  # (num_steps, num_grippers, num_points, 4)

            # add gripper pos
            g_pts = g_pts + gripper_center[:, :, None]

            # reshape
            g_pts = g_pts.reshape(n_steps, -1, 3)
            g_colors = g_colors.repeat(n_grippers, 1)
            g_quats = g_quats.reshape(n_steps, -1, 4)
            g_opacities = g_opacities.repeat(n_grippers, 1)
            g_scales = g_scales.repeat(n_grippers, 1)

            # merge
            bg_xyz = torch.cat([xyz, t_pts[None].repeat(n_steps, 1, 1), g_pts], dim=1)
            bg_rgb = torch.cat([rgb, t_colors[None].repeat(n_steps, 1, 1), g_colors[None].repeat(n_steps, 1, 1)], dim=1)
            bg_quat = torch.cat([quat, t_quats[None].repeat(n_steps, 1, 1), g_quats], dim=1)
            bg_opa = torch.cat([opa, t_opacities[None].repeat(n_steps, 1, 1), g_opacities[None].repeat(n_steps, 1, 1)], dim=1)
            bg_scales = torch.cat([scales, t_scales[None].repeat(n_steps, 1, 1), g_scales[None].repeat(n_steps, 1, 1)], dim=1)

            bg_quat = torch.nn.functional.normalize(bg_quat, dim=-1)

        rendervar_list = []
        rendervar_list_bg = []
        for t in range(n_steps):
            rendervar = {
                'means3D': xyz[t],  
                'colors_precomp': rgb[t],
                'rotations': quat[t],
                'opacities': opa[t],
                'scales': scales[t],
                'means2D': torch.zeros_like(xyz[t]),
            }
            rendervar_list.append(rendervar)

            if with_bg:
                rendervar_bg = {
                    'means3D': bg_xyz[t],  
                    'colors_precomp': bg_rgb[t],
                    'rotations': bg_quat[t],
                    'opacities': bg_opa[t],
                    'scales': bg_scales[t],
                    'means2D': torch.zeros_like(bg_xyz[t]),
                }
                rendervar_list_bg.append(rendervar_bg)

        return rendervar_list, rendervar_list_bg


def inverse_preprocess(cfg, p_x, grippers, source_data_root_episode):
    dx = cfg.sim.num_grids[-1]

    xyz_orig = np.load(source_data_root_episode / 'traj.npz')['xyz']
    xyz = torch.tensor(xyz_orig, dtype=torch.float32)
    
    R = torch.tensor(
        [[1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]]
    ).to(xyz.device).to(xyz.dtype)
    xyz = torch.einsum('nij,jk->nik', xyz, R.T)

    scale = cfg.sim.preprocess_scale
    xyz = xyz * scale
    
    if cfg.sim.preprocess_with_table:
        global_translation = torch.tensor([
            0.5 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
            dx * (cfg.model.clip_bound + 0.5) + 1e-5 - xyz[:, :, 1].min(),
            0.5 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
        ], dtype=xyz.dtype)
    else:
        global_translation = torch.tensor([
            0.5 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
            0.5 - (xyz[:, :, 1].max() + xyz[:, :, 1].min()) / 2,
            0.5 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
        ], dtype=xyz.dtype)

    p_x -= global_translation
    grippers[:, :, :3] -= global_translation

    p_x = p_x / scale
    grippers[:, :, :3] = grippers[:, :, :3] / scale

    p_x = torch.einsum('nij,jk->nik', p_x, torch.linalg.inv(R).T)
    grippers[:, :, :3] = torch.einsum('nmi,ik->nmk', grippers[:, :, :3], torch.linalg.inv(R).T)

    gripper_quat = grippers[:, :, 6:10]  # (n_steps, n_grippers, 4)
    gripper_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(gripper_quat)  # (n_steps, n_gripper, 3, 3)
    gripper_rot = R.T @ gripper_rot @ R
    gripper_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(gripper_rot)
    grippers[:, :, 6:10] = gripper_quat

    return p_x, grippers


def get_camera(cfg, log_root, source_data_dir, source_episode_id, frame_id=0, camera_id=1):
    h, w = 480, 848
    calibration_dir = (log_root.parent.parent / source_data_dir).parent / f'episode_{source_episode_id:04d}' / 'calibration'
    intr = np.load(calibration_dir / 'intrinsics.npy')
    rvec = np.load(calibration_dir / 'rvecs.npy')
    tvec = np.load(calibration_dir / 'tvecs.npy')
    R = [cv2.Rodrigues(rvec[i])[0] for i in range(rvec.shape[0])]
    T = [tvec[i, :, 0] for i in range(tvec.shape[0])]
    extrs = np.zeros((len(R), 4, 4)).astype(np.float32)
    for i in range(len(R)):
        extrs[i, :3, :3] = R[i]
        extrs[i, :3, 3] = T[i]
        extrs[i, 3, 3] = 1
    return {
        'w': w,
        'h': h,
        'intr': intr[camera_id],
        'w2c': extrs[camera_id],
    }


@torch.no_grad()
def render(
    cfg,
    log_root,
    iteration,
    episode_names,
    eval_dirname='eval',
    eval_postfix='',
    dataset_name='',
    camera_id=1,
    with_bg=False,
    with_mask=False,
    transparent=True,
    start_step=None,
    end_step=None,
):

    if dataset_name == '':
        eval_name = f'{cfg.train.name}/{eval_dirname}/{iteration:06d}'
    else:
        eval_name = f'{cfg.train.name}/{eval_dirname}/{dataset_name}/{iteration:06d}'
    render_type = 'pv_gs'
    render_type_gs = 'gs'

    exp_root: Path = log_root / eval_name
    state_root: Path = exp_root / 'state'
    image_root: Path = exp_root / render_type
    gs_root: Path = exp_root / render_type_gs
    mkdir(image_root, overwrite=cfg.overwrite, resume=cfg.resume)
    mkdir(gs_root, overwrite=cfg.overwrite, resume=cfg.resume)

    if with_mask:
        render_type_mask = 'mask'
        episode_mask_root = exp_root / render_type_mask
        mkdir(episode_mask_root, overwrite=cfg.overwrite, resume=cfg.resume)

    if with_bg:
        render_type_bg = 'pv_gs_bg'
        render_type_gs_bg = 'gs_bg'
        image_root_bg: Path = exp_root / render_type_bg
        gs_root_bg: Path = exp_root / render_type_gs_bg
        mkdir(image_root_bg, overwrite=cfg.overwrite, resume=cfg.resume)
        mkdir(gs_root_bg, overwrite=cfg.overwrite, resume=cfg.resume)

    video_path_list = []
    for episode_idx, episode in enumerate(episode_names):

        renderer = GSRenderer(cfg.render)

        meta = np.loadtxt(log_root / str(cfg.train.source_dataset_name) / episode / 'meta.txt')
        with open(log_root / str(cfg.train.source_dataset_name) / 'metadata.json') as f:
            datadir_list = json.load(f)
        episode_real_name = int(episode.split('_')[1])
        datadir = datadir_list[episode_real_name]
        source_data_dir = datadir['path']
        source_episode_id = int(meta[0])
        source_frame_start = int(meta[1]) + int(cfg.sim.n_history) * int(cfg.train.dataset_load_skip_frame) * int(cfg.train.dataset_skip_frame)
        source_frame_end = int(meta[2])
        episode_gs_init_path = (log_root.parent.parent / source_data_dir).parent / f'episode_{source_episode_id:04d}' / 'gs' / f'{source_frame_start:06d}.splat'

        renderer.load_params(episode_gs_init_path)

        episode_state_root = state_root / episode
        episode_image_root = image_root / episode
        episode_gs_root = gs_root / episode
        mkdir(episode_image_root, overwrite=cfg.overwrite, resume=cfg.resume)
        mkdir(episode_gs_root, overwrite=cfg.overwrite, resume=cfg.resume)

        if with_mask:
            episode_mask_root = episode_mask_root / episode
            mkdir(episode_mask_root, overwrite=cfg.overwrite, resume=cfg.resume)

        if with_bg:
            episode_image_root_bg = image_root_bg / episode
            episode_gs_root_bg = gs_root_bg / episode
            mkdir(episode_image_root_bg, overwrite=cfg.overwrite, resume=cfg.resume)
            mkdir(episode_gs_root_bg, overwrite=cfg.overwrite, resume=cfg.resume)

        ckpt_paths = list(sorted(episode_state_root.glob('*.pt'), key=lambda x: int(x.stem)))

        p_x_list = []
        grippers_list = []
        for i, path in enumerate(ckpt_paths):
            if i % cfg.render.skip_frame != 0:
                continue

            ckpt = torch.load(path, map_location='cpu')
            p_x = ckpt['x']
            p_x_list.append(p_x)

            use_grippers = 'grippers' in ckpt
            grippers = None
            if use_grippers:
                grippers = ckpt['grippers']
            if grippers is not None:
                grippers_list.append(grippers)
        
        p_x_list = torch.stack(p_x_list, dim=0)
        grippers_list = torch.stack(grippers_list, dim=0) if grippers_list else []
        p_x_list, grippers_list = inverse_preprocess(cfg, p_x_list, grippers_list,
                source_data_root_episode=log_root / cfg.train.source_dataset_name / episode)

        rendervar_list, rendervar_list_bg = renderer.rollout_and_render(p_x_list, grippers_list, with_bg=with_bg)

        for i, path in enumerate(tqdm(ckpt_paths, desc=render_type)):
            rendervar = rendervar_list[i // cfg.render.skip_frame]
            renderer.set_camera(**get_camera(cfg, log_root, source_data_dir, source_episode_id, frame_id=i, camera_id=camera_id))
            im, _ = renderer.render(rendervar, 0)
            im = im.cpu().numpy().transpose(1, 2, 0)
            im = (im * 255).astype(np.uint8)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            if transparent or with_mask:
                rendervar['colors_precomp'] = torch.ones_like(rendervar['colors_precomp'])
                mask, _ = renderer.render(rendervar, 0)
                mask = mask.cpu().numpy().transpose(1, 2, 0)

                if transparent:
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
                    im[:, :, 3] = (mask * 255).mean(-1).astype(np.uint8)

                if with_mask:
                    thresh = 0.1
                    mask = (mask > thresh).astype(np.float32)
                    mask = (mask * 255).astype(np.uint8)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    save_path = str(episode_mask_root / f'{i // cfg.render.skip_frame:04d}.png')
                    cv2.imwrite(save_path, mask)

            save_path = str(episode_image_root / f'{i // cfg.render.skip_frame:04d}.png')
            cv2.imwrite(save_path, im)

            gs_save_path = str(episode_gs_root / f'{i // cfg.render.skip_frame:04d}.splat')
            save_to_splat(
                pts=rendervar['means3D'].cpu().numpy(),
                colors=rendervar['colors_precomp'].cpu().numpy(),
                scales=rendervar['scales'].cpu().numpy(),
                quats=rendervar['rotations'].cpu().numpy(),
                opacities=rendervar['opacities'].cpu().numpy(),
                output_file=gs_save_path,
                center=False,
                rotate=False,
            )

            if with_bg:
                rendervar_bg = rendervar_list_bg[i // cfg.render.skip_frame]
                renderer.set_camera(**get_camera(cfg, log_root, source_data_dir, source_episode_id, frame_id=i, camera_id=camera_id))
                im, _ = renderer.render(rendervar_bg, 0)
                im = im.cpu().numpy().transpose(1, 2, 0)
                im = (im * 255).astype(np.uint8)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                
                if transparent:
                    rendervar_bg['colors_precomp'] = torch.ones_like(rendervar_bg['colors_precomp'])
                    mask, _ = renderer.render(rendervar_bg, 0)
                    mask = mask.cpu().numpy().transpose(1, 2, 0)
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
                    im[:, :, 3] = (mask * 255).mean(-1).astype(np.uint8)

                save_path = str(episode_image_root_bg / f'{i // cfg.render.skip_frame:04d}.png')
                cv2.imwrite(save_path, im)

                gs_save_path = str(episode_gs_root_bg / f'{i // cfg.render.skip_frame:04d}.splat')
                save_to_splat(
                    pts=rendervar_bg['means3D'].cpu().numpy(),
                    colors=rendervar_bg['colors_precomp'].cpu().numpy(),
                    scales=rendervar_bg['scales'].cpu().numpy(),
                    quats=rendervar_bg['rotations'].cpu().numpy(),
                    opacities=rendervar_bg['opacities'].cpu().numpy(),
                    output_file=gs_save_path,
                    center=False,
                    rotate=False,
                )

        make_video(episode_image_root, image_root / f'{episode}{eval_postfix}.mp4', '%04d.png', cfg.render.fps)
        video_path_list.append(image_root / f'{episode}{eval_postfix}.mp4')
    
        if with_bg:
            make_video(episode_image_root_bg, image_root_bg / f'{episode}{eval_postfix}.mp4', '%04d.png', cfg.render.fps)
    
    return video_path_list


@torch.no_grad()
def do_gs(*args, **kwargs):
    ret = render(*args, **kwargs)
    return ret
