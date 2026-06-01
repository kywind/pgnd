import numpy as np
from pathlib import Path
from copy import deepcopy
from functools import partial
import torch
import torch.nn as nn
import warp as wp
from dgl.geometry import farthest_point_sampler
import random
import kornia
import open3d as o3d
from tqdm import tqdm, trange

from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, MPMStaticsBatch, MPMCollidersBatch
from pgnd.utils import get_root, mkdir
from pgnd.ffmpeg import make_video

from train_eval import transform_gripper_points
from gs.convert import read_splat

root: Path = get_root(__file__)


def fps(x, n, device, random_start=False):
    start_idx = random.randint(0, x.shape[0] - 1) if random_start else 0
    fps_idx = farthest_point_sampler(x[None], n, start_idx=start_idx)[0]
    fps_idx = fps_idx.to(x.device)
    return fps_idx


class DynamicsModule:

    def __init__(self, cfg, exp_root, ckpt_path, batch_size, num_steps_total):

        wp.init()
        wp.ScopedTimer.enabled = False
        wp.set_module_options({'fast_math': False})
        wp.config.verify_autograd_array_access = True

        self.exp_root = exp_root
        self.batch_size = batch_size
        self.num_steps_total = num_steps_total

        gpus = [int(gpu) for gpu in cfg.gpus]
        self.gpus = gpus

        wp_devices = [wp.get_device(f'cuda:{gpu}') for gpu in gpus]
        torch_devices = [torch.device(f'cuda:{gpu}') for gpu in gpus]
        device_count = len(torch_devices)
        assert device_count == 1
        wp_device = wp_devices[0]
        torch_device = torch_devices[0]
        self.device = torch_device
        
        self.cfg = cfg
        n_history = cfg.sim.n_history
        material: nn.Module = getattr(pgnd.material, cfg.env.blob.material.material.cls)(cfg.env.blob.material.material, n_history)
        material.set_params(num_grids=cfg.sim.num_grids)
        material.to(torch_device)
        material.requires_grad_(False)
        material.train(False)

        friction: nn.Module = Friction(np.array([cfg.sim.friction])[None].repeat(batch_size, axis=0))  # type: ignore
        friction.to(torch_device)
        assert len(list(friction.parameters())) > 0
        friction.requires_grad_(False)
        friction.train(False)

        ckpt = torch.load(ckpt_path, map_location=torch_device)
        material.load_state_dict(ckpt['material'])
        self.material = material
        self.friction = friction

        self.material.eval()
        self.friction.eval()

        self.preprocess_metadata = {}
        self.downsample_indices = None
        self.target_state = None

        if cfg.sim.gripper_points:  # do some manual transformation here
            pts, colors, scales, quats, opacities = read_splat('experiments/log/gs/ckpts/gripper_new.splat')
            n_gripper_particles = 500
            R = np.array(
                [[1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]]
            )
            eef_global_T = np.array([cfg.env.blob.eef_t[0], cfg.env.blob.eef_t[1], cfg.env.blob.eef_t[2] - 0.01])  # 1018_sloth: 0.185, 1018_rope_short: 0.013)
            pts = pts + eef_global_T
            pts = pts @ R.T
            scale = cfg.sim.preprocess_scale
            pts = pts * scale

            axis = np.array([0, 1, 0])
            angle = -27  # degree
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi / 180 * angle)
            translation = np.array([-0.015, 0.06, 0])

            pts = pts @ R.T
            pts = pts + translation

            R = np.array(
                [[0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0]]
            )
            pts = pts @ R.T

            gripper_pts = torch.from_numpy(pts).to(torch.float32).to(self.device)
            downsample_indices = fps(gripper_pts, n_gripper_particles, self.device, random_start=True)
            gripper_pts = gripper_pts[downsample_indices]
            self.gripper_pts = gripper_pts

    def set_target_state(self, target_state):
        self.target_state = target_state

    def reset_model(self, x=None):
        return
    
    def reset_preprocess_meta(self, pts):
        cfg = self.cfg
        dx = cfg.sim.num_grids[-1]

        p_x = torch.tensor(pts).to(torch.float32).to(self.device)
        R = torch.tensor(
            [[1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]]
        ).to(p_x.device).to(p_x.dtype)
        p_x_rotated = p_x @ R.T

        scale = cfg.sim.preprocess_scale
        p_x_rotated_scaled = p_x_rotated * scale

        cfg = self.cfg
        if cfg.sim.preprocess_with_table:
            global_translation = torch.tensor([
                0.5 - (p_x_rotated_scaled[:, 0].max() + p_x_rotated_scaled[:, 0].min()) / 2,
                dx * (cfg.env.blob.clip_bound + 0.5) + 1e-5 - p_x_rotated_scaled[:, 1].min(),
                0.5 - (p_x_rotated_scaled[:, 2].max() + p_x_rotated_scaled[:, 2].min()) / 2,
            ], dtype=p_x_rotated_scaled.dtype, device=p_x_rotated_scaled.device)
        else:
            global_translation = torch.tensor([
                0.5 - (p_x_rotated_scaled[:, 0].max() + p_x_rotated_scaled[:, 0].min()) / 2,
                0.5 - (p_x_rotated_scaled[:, 1].max() + p_x_rotated_scaled[:, 1].min()) / 2,
                0.5 - (p_x_rotated_scaled[:, 2].max() + p_x_rotated_scaled[:, 2].min()) / 2,
            ], dtype=p_x_rotated_scaled.dtype, device=p_x_rotated_scaled.device)

        self.preprocess_metadata = {
            'R': R,
            'scale': scale,
            'global_translation': global_translation,
        }
    
    def reset_downsample_indices(self, pts, uniform=True):
        cfg = self.cfg
        if uniform:
            downsample_indices = fps(pts, cfg.sim.n_particles, self.device, random_start=True)
        else:
            downsample_indices = torch.randperm(pts.shape[0])[:cfg.sim.n_particles]
        self.downsample_indices = downsample_indices

    def rollout(self, pts, eef_xyz, eef_rot, eef_gripper, pts_his=None):
        cfg = self.cfg

        # preprocess eef
        # eef_xyz: (batch_size, n_look_forward + 1, n_grippers, 3)
        # eef_rot: (batch_size, n_look_forward + 1, n_grippers, 3, 3)
        # eef_gripper: (batch_size, n_look_forward + 1, n_grippers, 1)  0: close, 1: open
        batch_size = eef_xyz.shape[0]
        assert eef_xyz.shape[1] == eef_rot.shape[1] == eef_gripper.shape[1]

        R = self.preprocess_metadata['R']
        scale = self.preprocess_metadata['scale']
        global_translation = self.preprocess_metadata['global_translation']

        eef_xyz = eef_xyz @ R.T
        eef_xyz = eef_xyz * scale
        eef_xyz += global_translation

        eef_rot = eef_rot @ R.T
        eef_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot)

        n_frames = eef_xyz.shape[1] - 1
        eef_vel = torch.zeros_like(eef_xyz[:, 1:])  # (batch_size, n_frames, n_grippers, 3)
        eef_vel = (eef_xyz[:, 1:] - eef_xyz[:, :-1]) / cfg.sim.dt

        eef_rot_this = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat[:, :-1].reshape(-1, 4))  # (batch_size * n_frames * n_grippers, 3, 3)
        eef_rot_next = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat[:, 1:].reshape(-1, 4))  # (batch_size * n_frames * n_grippers, 3, 3)
        eef_rot_delta = eef_rot_this.bmm(eef_rot_next.inverse())
        eef_aa = kornia.geometry.conversions.rotation_matrix_to_axis_angle(eef_rot_delta)  # (batch_size * n_frames * n_grippers, 3)

        eef_quat_vel = torch.zeros((batch_size, n_frames, cfg.sim.num_grippers, 3)).to(self.device).to(torch.float32)
        eef_quat_vel = eef_aa.reshape(batch_size, n_frames, -1, 3) / cfg.sim.dt  # (batch_size, n_frames, n_grippers, 3), radian per second

        grippers = torch.zeros((batch_size, n_frames, cfg.sim.num_grippers, 15)).to(self.device).to(torch.float32)
        grippers[:, :, :, :3] = eef_xyz[:, :-1]  # not using the last position although we gonna arrive there
        grippers[:, :, :, 3:6] = eef_vel
        grippers[:, :, :, 6:10] = eef_quat[:, :-1]
        grippers[:, :, :, 10:13] = eef_quat_vel
        grippers[:, :, :, 13] = cfg.env.blob.gripper_radius
        grippers[:, :, :, 14] = eef_gripper[:, :-1].squeeze(-1)

        # preprocess pts
        x = pts[self.downsample_indices]

        R = self.preprocess_metadata['R']
        scale = self.preprocess_metadata['scale']
        global_translation = self.preprocess_metadata['global_translation']

        # data frame to model frame
        x = x @ R.T
        x = x * scale
        x = x + global_translation

        x = x[None].repeat(batch_size, 1, 1)
        x_pred, v_pred = self.rollout_preprocessed(x, grippers=grippers)  # assumes static
        
        # inverse preprocess
        x_pred = x_pred - global_translation
        x_pred = x_pred / scale
        x_pred = x_pred @ R

        v_pred = v_pred / scale
        v_pred = v_pred @ R
        return x_pred, v_pred

    @torch.no_grad()
    def rollout_preprocessed(self, x, v=None, x_his=None, v_his=None, grippers=None):
        cfg = self.cfg

        # reset model
        wp_devices = [wp.get_device(f'cuda:{gpu}') for gpu in self.gpus]
        torch_devices = [torch.device(f'cuda:{gpu}') for gpu in self.gpus]
        device_count = len(torch_devices)
        assert device_count == 1
        wp_device = wp_devices[0]
        torch_device = torch_devices[0]

        batch_size = x.shape[0]
        num_particles = x.shape[1]
        assert num_particles == cfg.sim.n_particles
        assert batch_size == self.batch_size

        clip_bound = torch.tensor(cfg.env.blob.clip_bound)

        if cfg.sim.gripper_points:
            gripper_points = self.gripper_pts.clone()[None].repeat(batch_size, 1, 1)  # (bsz, num_grippers, 3)
            gripper_x, gripper_v, gripper_mask = transform_gripper_points(cfg, gripper_points, grippers)  # (bsz, num_steps, num_grippers, 3)
            num_gripper_particles = gripper_x.shape[2]
            num_particles_orig = num_particles
            num_particles = num_particles + num_gripper_particles
            num_grippers = 0
        else:
            gripper_x = None
            gripper_v = None
            gripper_mask = None
            num_particles_orig = num_particles
            num_gripper_particles = 0
            num_grippers = cfg.sim.num_grippers

        sim = CacheDiffSimWithFrictionBatch(cfg, num_steps_total, batch_size, self.wp_device, requires_grad=True)

        statics = StaticsBatch()
        statics.init(shape=(batch_size, num_particles), device=self.wp_device)
        statics.update_clip_bound(clip_bound)
        colliders = CollidersBatch()
        colliders.init(shape=(batch_size, num_grippers), device=wp_device)

        if v is None:
            v = torch.zeros_like(x)

        if cfg.sim.n_history > 0:
            if x_his is None:
                x_his = x.clone().repeat(1, 1, cfg.sim.n_history)
            if v_his is None:
                v_his = v.clone().repeat(1, 1, cfg.sim.n_history)

        colliders.initialize_grippers(grippers[:, 0])

        assert cfg.sim.skip_frame == cfg.sim.interval  # required for the following line

        xs = []
        vs = []
        for step in trange(self.num_steps_total):
            colliders.update_grippers(grippers[:, step])  # ignore gripper radius
            x_in = x.clone()
            v_in = v.clone()

            if cfg.sim.gripper_points:
                assert gripper_x is not None and gripper_v is not None and gripper_mask is not None
                x = torch.cat([x, gripper_x[:, step]], dim=1)  # gripper_x: (bsz, num_steps, num_particles, 3)
                v = torch.cat([v, gripper_v[:, step]], dim=1)
                x_his = torch.cat([x_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=x_his.device, dtype=x_his.dtype)], dim=1)  # type: ignore
                v_his = torch.cat([v_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=v_his.device, dtype=v_his.dtype)], dim=1)  # type: ignore
                
                if C.shape[1] < num_particles:
                    C = torch.cat([C, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], 3, 3), device=C.device, dtype=C.dtype)], dim=1)
                if F.shape[1] < num_particles:
                    F = torch.cat([F, torch.eye(3, device=F.device).unsqueeze(0).unsqueeze(0).repeat(gripper_x.shape[0], gripper_x.shape[2], 1, 1)], dim=1)
                if enabled.shape[1] < num_particles:
                    enabled = torch.cat([enabled, gripper_mask[:, step]], dim=1)
                statics.update_enabled(enabled.cpu())

            pred = self.material(x, v, x_his, v_his, C, F)

            if pred.isnan().any():
                print('pred isnan', pred.min().item(), pred.max().item())
                break
            if pred.isinf().any():
                print('pred isinf', pred.min().item(), pred.max().item())
                break

            x, v = sim(statics, colliders, step, x, v, self.friction.mu, pred)

            if cfg.sim.gripper_forcing:
                assert not cfg.sim.gripper_points
                assert grippers is not None and x_in is not None
                gripper_xyz = grippers[:, step, :, :3]  # (bsz, num_grippers, 3)
                gripper_v = grippers[:, step, :, 3:6]  # (bsz, num_grippers, 3)
                x_from_gripper = x_in[:, None] - gripper_xyz[:, :, None]  # (bsz, num_grippers, num_particles, 3)
                x_gripper_distance = torch.norm(x_from_gripper, dim=-1)  # (bsz, num_grippers, num_particles)
                x_gripper_distance_mask = x_gripper_distance < cfg.env.blob.gripper_radius
                x_gripper_distance_mask = x_gripper_distance_mask.unsqueeze(-1).repeat(1, 1, 1, 3)  # (bsz, num_particles, num_grippers, 3)
                gripper_v_expand = gripper_v[:, :, None].repeat(1, 1, num_particles, 1)  # (bsz, num_grippers, num_particles, 3)

                gripper_closed = grippers[:, step, :, -1] < 0.5  # (bsz, num_grippers)  # 1: open, 0: close
                x_gripper_distance_mask = torch.logical_and(x_gripper_distance_mask, gripper_closed[:, :, None, None].repeat(1, 1, num_particles, 3))

                gripper_quat_vel = grippers[:, step, :, 10:13]  # (bsz, num_grippers, 3)
                gripper_angular_vel = torch.norm(gripper_quat_vel, dim=-1, keepdim=True)  # (bsz, num_grippers, 1)
                gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)  # (bsz, num_grippers, 3)

                grid_from_gripper_axis = x_from_gripper - \
                    (gripper_quat_axis[:, :, None] * x_from_gripper).sum(dim=-1, keepdim=True) * gripper_quat_axis[:, :, None]  # (bsz, num_grippers, num_particles, 3)
                gripper_v_expand = torch.cross(gripper_quat_vel[:, :, None], grid_from_gripper_axis, dim=-1) + gripper_v_expand

                for i in range(gripper_xyz.shape[1]):
                    x_gripper_distance_mask_single = x_gripper_distance_mask[:, i]
                    x[x_gripper_distance_mask_single] = x_in[x_gripper_distance_mask_single] + cfg.sim.dt * gripper_v_expand[:, i][x_gripper_distance_mask_single]
                    v[x_gripper_distance_mask_single] = gripper_v_expand[:, i][x_gripper_distance_mask_single]
            
            if cfg.sim.n_history > 0:
                assert x_his is not None and v_his is not None
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
                x = x[:, :num_particles_orig]
                v = v[:, :num_particles_orig]
                enabled = enabled[:, :num_particles_orig]

            colliders_save = colliders.export()
            colliders_save = {key: torch.from_numpy(colliders_save[key])[0].to(x.device).to(x.dtype) for key in colliders_save}

            xs.append(x.detach().clone())
            vs.append(v.detach().clone())

        xs = torch.stack(xs, dim=1)  # (batch_size, num_steps_total, num_particles, 3)
        vs = torch.stack(vs, dim=1)  # (batch_size, num_steps_total, num_particles, 3)

        return xs, vs
