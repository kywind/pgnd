from pathlib import Path
from typing import Union, Optional
from omegaconf import DictConfig

import os
import torch
import time
import shutil
import json
import yaml
import random
import glob
import kornia
import numpy as np
import pickle as pkl
import open3d as o3d
from torch.utils.data import Dataset
from dgl.geometry import farthest_point_sampler
from sklearn.neighbors import NearestNeighbors



def preprocess(cfg, dataset_root_episode, source_data_root_episode, dt_frame, dx):
    print('Preprocessing episode:', dataset_root_episode.name)
    assert dt_frame - 1. / 30. < 1e-3

    # transform xyz: x -> x, z -> -y, y -> z
    R = np.array(
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]]
    )

    eef_global_T = np.array([cfg.model.eef_t[0], cfg.model.eef_t[1], cfg.model.eef_t[2]])  # 1018_sloth: 0.185, 1018_rope_short: 0.01

    xyz = np.load(source_data_root_episode / 'traj.npz')['xyz']
    eef_xyz = np.loadtxt(source_data_root_episode / 'eef_traj.txt')
    n_frames = eef_xyz.shape[0]
    eef_xyz = eef_xyz.reshape(eef_xyz.shape[0], -1, 3)  # (n_frames, n_grippers, 3)
    eef_xyz = eef_xyz + eef_global_T

    xyz = np.einsum('nij,jk->nik', xyz, R.T)
    eef_xyz = np.einsum('nij,jk->nik', eef_xyz, R.T)  # (n_frames, n_grippers, 3)

    if os.path.exists(source_data_root_episode / 'eef_rot.txt'):
        eef_rot = np.loadtxt(source_data_root_episode / 'eef_rot.txt')  # (n_frames, -1)
        eef_rot = eef_rot.reshape(eef_rot.shape[0], -1, 3, 3)  # (n_frames, n_grippers, 3, 3)
        eef_rot = R @ eef_rot @ R.T
        eef_rot = torch.from_numpy(eef_rot).reshape(-1, 3, 3)  # (n_frames, n_grippers, 3, 3)
        eef_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot).numpy().reshape(n_frames, -1, 4)  # (n_frames, n_grippers, 4)
    else:
        eef_quat = np.zeros((xyz.shape[0], cfg.sim.num_grippers, 4), dtype=np.float32)  # (n_frames, n_grippers, 4)
        eef_quat[:, :, 0] = 1.0  # (n_frames, n_grippers, 4)

    if os.path.exists(source_data_root_episode / 'eef_gripper.txt'):
        eef_gripper = np.loadtxt(source_data_root_episode / 'eef_gripper.txt')  # (n_frames, n_grippers)
        eef_gripper = eef_gripper.reshape(eef_gripper.shape[0], -1)
        assert cfg.sim.num_grippers == eef_gripper.shape[1]
    else:
        eef_gripper = np.zeros((n_frames, cfg.sim.num_grippers), dtype=np.float32)  # (n_frames, n_grippers)
    assert cfg.sim.num_grippers == eef_quat.shape[1]

    scale = cfg.sim.preprocess_scale
    xyz = xyz * scale
    eef_xyz = eef_xyz * scale

    n_frames = xyz.shape[0]

    # construct colliders: position, velocity, height, radius, axis
    eef_vel = np.zeros_like(eef_xyz)  # (n_frames, n_grippers, 3)
    eef_vel[:-1] = (eef_xyz[1:] - eef_xyz[:-1]) / dt_frame
    eef_vel[-1] = eef_vel[-2]

    # construct rotational velocity
    eef_rot_this = kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.from_numpy(eef_quat[:-1]).reshape(-1, 4))  # (n_frames-1 * n_grippers, 3, 3)
    eef_rot_next = kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.from_numpy(eef_quat[1:]).reshape(-1, 4))  # (n_frames-1 * n_grippers, 3, 3)
    eef_rot_delta = eef_rot_this.bmm(eef_rot_next.inverse())
    eef_aa = kornia.geometry.conversions.rotation_matrix_to_axis_angle(eef_rot_delta)  # (n_frames-1 * n_grippers, 3)

    eef_quat_vel = np.zeros((n_frames, cfg.sim.num_grippers, 3), dtype=eef_quat.dtype)
    eef_quat_vel[:-1] = eef_aa.reshape(n_frames - 1, -1, 3) / dt_frame  # (n_frames-1, n_grippers, 3), radian per second
    eef_quat_vel[-1] = eef_quat_vel[-2]

    grippers = np.zeros((n_frames, cfg.sim.num_grippers, 15))
    grippers[:, :, :3] = eef_xyz
    grippers[:, :, 3:6] = eef_vel
    grippers[:, :, 6:10] = eef_quat
    grippers[:, :, 10:13] = eef_quat_vel
    grippers[:, :, 13] = cfg.model.gripper_radius
    grippers[:, :, 14] = eef_gripper
    
    if cfg.sim.preprocess_with_table:  # drop everything to the table plane (y=0)
        global_translation = np.array([
            0.5 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
            dx * (cfg.model.clip_bound + 0.5) + 1e-5 - xyz[:, :, 1].min(),
            0.5 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
        ], dtype=xyz.dtype)
    else:
        global_translation = np.array([
            0.5 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
            0.5 - (xyz[:, :, 1].max() + xyz[:, :, 1].min()) / 2,
            0.5 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
        ], dtype=xyz.dtype)
    xyz += global_translation
    grippers[:, :, :3] += global_translation
    
    if not (xyz[:, :, 0].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
        and xyz[:, :, 0].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6 \
        and xyz[:, :, 1].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
        and xyz[:, :, 1].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6 \
        and xyz[:, :, 2].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
        and xyz[:, :, 2].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6):
        print(dataset_root_episode.name, 'out of bound')
        xyz_max = xyz.max(axis=0)
        xyz_max_mask = (xyz_max[:, 0] > 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6) | \
            (xyz_max[:, 1] > 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6) | \
            (xyz_max[:, 2] > 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6)
        xyz_min = xyz.min(axis=0)
        xyz_min_mask = (xyz_min[:, 0] < dx * (cfg.model.clip_bound + 0.5) - 1e-6) | \
            (xyz_min[:, 1] < dx * (cfg.model.clip_bound + 0.5) - 1e-6) | \
            (xyz_min[:, 2] < dx * (cfg.model.clip_bound + 0.5) - 1e-6)
        xyz_mask = xyz_max_mask | xyz_min_mask
        xyz = xyz[:, ~xyz_mask]
        
        assert (xyz[:, :, 0].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
            and xyz[:, :, 0].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6 \
            and xyz[:, :, 1].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
            and xyz[:, :, 1].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6 \
            and xyz[:, :, 2].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
            and xyz[:, :, 2].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6)

    data = {
        'num_particles': xyz.shape[1],
        'clip_bound': cfg.model.clip_bound,
    }
    with open(dataset_root_episode / 'info.yaml', 'w') as f:
        yaml.dump(data, f)
    
    # save data
    for frame_id in range(n_frames):
        data = {
            'x': torch.tensor(xyz[frame_id]).float(),
            'grippers': torch.tensor(grippers[frame_id]).float()
        }
        torch.save(data, dataset_root_episode / f'{frame_id:04d}.pt')
    return True  # necessary for success check


def fps(x, enabled, n, device, random_start=False):
    assert torch.diff(enabled * 1.0).sum() in [0.0, -1.0]
    start_idx = random.randint(0, enabled.sum() - 1) if random_start else 0
    fps_idx = farthest_point_sampler(x[enabled][None], n, start_idx=start_idx)[0]
    fps_idx = fps_idx.to(x.device)
    return fps_idx


class RealTeleopBatchDataset(Dataset):

    def __init__(self,
                cfg: DictConfig, 
                dataset_root: Union[str, Path],
                source_data_root: Union[str, Path], 
                device: torch.device, 
                num_steps: int, 
                train=False, 
                eval_episode_name=None, 
                dataset_non_overwrite=None
        ) -> None:
        super().__init__()
        self.device = device
        dataset_root = Path(dataset_root).resolve()
        dataset_root.mkdir(parents=True, exist_ok=True)
        self.root = dataset_root
        self.is_train = train

        self.interval = cfg.sim.interval
        self.dt = eval(cfg.sim.dt) if isinstance(cfg.sim.dt, str) else cfg.sim.dt
        self.num_steps = num_steps
        self.skip_frame = cfg.train.dataset_skip_frame
        self.n_particles = cfg.sim.n_particles
        self.n_history = cfg.sim.n_history
        self.uniform = cfg.sim.uniform or not train  # always use uniform sampling during test
        print('Using uniform sampling:', self.uniform)
        self.load_skip_frame = cfg.train.dataset_load_skip_frame
        self.dx = cfg.sim.num_grids[-1]

        self.return_downsample_indices = (eval_episode_name is not None)

        with torch.no_grad():
            source_data_root = Path(source_data_root).resolve()
            episodes = list(sorted(Path(str(source_data_root)).glob('episode_*')))
            if len(episodes) == 0:
                import ipdb; ipdb.set_trace()
            if eval_episode_name is not None:
                assert not train
                episodes = [e for e in episodes if e.name == eval_episode_name]
            if train:
                episodes = episodes[cfg.train.training_start_episode:cfg.train.training_end_episode]

            episodes_clean = []
            episodes_xs = []
            episodes_vs = []
            episodes_enabled = []
            episodes_grippers = []
            episodes_clip_bound = []
            len_sim_list = []
            traj_list = []
            meta_list = []
            use_grippers = False
            max_num_particles = 0
            max_num_frames = 0
            overwrite = None if eval_episode_name is None else True

            for episode_id in range(len(episodes)):
                source_data_root_episode = episodes[episode_id]
                episode_name = source_data_root_episode.name
                dataset_root_episode = dataset_root / episode_name
                if os.path.exists(dataset_root_episode):
                    traj = list(sorted(dataset_root_episode.glob('*.pt')))[::self.load_skip_frame]
                    if len(traj) > 0:
                        if dataset_non_overwrite:
                            overwrite = False
                        while overwrite is None:
                            feedback = input(f'{dataset_root_episode} already exists, overwrite? [y/n] ')
                            ret = feedback.casefold()
                            if ret == 'y':
                                overwrite = True
                                break
                            elif ret == 'n':
                                overwrite = False
                                break
                        if overwrite == True: 
                            shutil.rmtree(dataset_root_episode, ignore_errors=True)
                        else:
                            assert overwrite == False
                            episodes_clean.append(episode_id)
                    elif overwrite is None: overwrite = True
                    elif overwrite == False: continue
                elif overwrite is None: overwrite = True
                elif overwrite == False: continue
                if overwrite:
                    dataset_root_episode.mkdir(parents=True, exist_ok=True)
                    success = preprocess(cfg, dataset_root_episode, source_data_root_episode,
                            self.dt * self.interval / self.skip_frame / self.load_skip_frame, self.dx)
                    if not success:
                        shutil.rmtree(dataset_root_episode, ignore_errors=True)
                        continue
                    else:
                        episodes_clean.append(episode_id)
                    traj = list(sorted(dataset_root_episode.glob('*.pt')))[::self.load_skip_frame]

                states = [torch.load(p, map_location='cpu') for p in traj]

                xs = torch.stack([state['x'] for state in states], dim=0)
                vs = torch.zeros_like(xs)
                vs[1:] = (xs[1:] - xs[:-1]) / (self.dt * self.interval / self.skip_frame) #  / self.load_skip_frame)
                vs[0] = vs[1]
                traj_list.append(traj)

                num_particles = xs.size(1)
                max_num_particles = max(max_num_particles, num_particles)
                max_num_frames = max(len(traj), max_num_frames)

                len_sim = max(1, len(traj) - (self.num_steps + self.n_history) * self.skip_frame)
                len_sim_list.append(len_sim)

                episodes_xs.append(xs)
                episodes_vs.append(vs)

                grippers = torch.stack([state['grippers'] for state in states], dim=0)
                episodes_grippers.append(grippers)

                # load yaml
                yaml_path = dataset_root_episode / 'info.yaml'
                with open(yaml_path, 'r') as f:
                    info = yaml.safe_load(f)
                clip_bound = info['clip_bound']
                episodes_clip_bound.append(clip_bound)
            
            for episode_id in range(len(episodes_clean)):
                xs = episodes_xs[episode_id]
                vs = episodes_vs[episode_id]
                enabled = torch.ones(xs.size(1), dtype=torch.bool)
                num_particles_pad = max_num_particles - xs.size(1)
                num_frames_pad = max_num_frames - xs.size(0)
                xs = torch.nn.functional.pad(xs, (0, 0, 0, num_particles_pad, 0, num_frames_pad), value=0)
                vs = torch.nn.functional.pad(vs, (0, 0, 0, num_particles_pad, 0, num_frames_pad), value=0)
                enabled = torch.nn.functional.pad(enabled, (0, num_particles_pad), value=0)
                episodes_xs[episode_id] = xs
                episodes_vs[episode_id] = vs
                episodes_enabled.append(enabled)
                
                grippers = episodes_grippers[episode_id]
                grippers = torch.nn.functional.pad(grippers, (0, 0, 0, 0, 0, num_frames_pad), value=0)
                episodes_grippers[episode_id] = grippers

            self.episode_xs = torch.stack(episodes_xs, dim=0)  # (n_episodes, n_frames, n_particles, 3)
            self.episode_vs = torch.stack(episodes_vs, dim=0)
            self.episode_enabled = torch.stack(episodes_enabled, dim=0)  # (n_episodes, n_particles)
            self.episode_grippers = torch.stack(episodes_grippers, dim=0)
            self.episode_clip_bound = torch.tensor(episodes_clip_bound, dtype=torch.float32)
            self.len_sim_list = np.array(len_sim_list)
            self.total_len_sim = sum(len_sim_list)
            self.max_num_particles = max_num_particles
            self.traj_list = traj_list
            self.meta_list = meta_list
        
        print('Finished loading dataset')
        print('Total number of episodes:', len(episodes_clean))
        print('Total number of frames:', self.total_len_sim)
        assert len(self.episode_xs) == len(episodes_clean)

    def __len__(self) -> int:
        return self.total_len_sim


    def __getitem__(self, index):
        cum_len_sim = np.cumsum(self.len_sim_list)
        episode = np.where(cum_len_sim - index > 0)[0][0]
        frame = index - cum_len_sim[episode - 1] if episode > 0 else index

        # history frames
        frame = frame + self.n_history * self.skip_frame

        x = self.episode_xs[episode][frame]
        v = self.episode_vs[episode][frame]
        enabled = self.episode_enabled[episode]

        # downsample
        if self.uniform:
            downsample_indices = fps(x, enabled, self.n_particles, self.device, random_start=True)
        else:
            downsample_indices = torch.randperm(enabled.sum())[:self.n_particles]
        x = x[downsample_indices]
        v = v[downsample_indices]

        x_his = torch.zeros((self.n_particles, 0), dtype=x.dtype, device=x.device)
        v_his = torch.zeros((self.n_particles, 0), dtype=v.dtype, device=v.device)
        for his_id in range(-self.n_history, 0):
            his_frame = frame + his_id * self.skip_frame
            assert his_frame >= 0
            x_his_frame = self.episode_xs[episode][his_frame]
            v_his_frame = self.episode_vs[episode][his_frame]
            x_his_frame = x_his_frame[downsample_indices]
            v_his_frame = v_his_frame[downsample_indices]
            x_his = torch.cat([x_his, x_his_frame], dim=-1)
            v_his = torch.cat([v_his, v_his_frame], dim=-1)

        enabled = torch.ones(x.size(0), dtype=torch.bool)
        clip_bound = self.episode_clip_bound[episode]
        episode_vec = torch.tensor([episode, frame])
        init_state = (x, v, x_his, v_his, clip_bound, enabled, episode_vec)

        end_frame = frame + self.num_steps * self.skip_frame + 1

        grippers_end = self.episode_grippers[episode][frame+self.skip_frame:end_frame:self.skip_frame]
        grippers_start = self.episode_grippers[episode][frame:end_frame-self.skip_frame:self.skip_frame][:grippers_end.shape[0]]
        step_dt = self.dt * self.interval
        actions = grippers_start.clone()
        actions[:, :, 3:6] = (grippers_end[:, :, :3] - grippers_start[:, :, :3]) / step_dt

        rot_start = kornia.geometry.conversions.quaternion_to_rotation_matrix(grippers_start[:, :, 6:10].reshape(-1, 4))
        rot_end = kornia.geometry.conversions.quaternion_to_rotation_matrix(grippers_end[:, :, 6:10].reshape(-1, 4))
        rot_delta = rot_start.bmm(rot_end.transpose(1, 2))  # same delta convention as preprocess and DynamicsModule.rollout
        aa = kornia.geometry.conversions.rotation_matrix_to_axis_angle(rot_delta)
        actions[:, :, 10:13] = aa.reshape(actions.shape[0], actions.shape[1], 3) / step_dt

        gt_xs = self.episode_xs[episode][frame+self.skip_frame:end_frame:self.skip_frame]
        gt_vs = self.episode_vs[episode][frame+self.skip_frame:end_frame:self.skip_frame]
        
        gt_xs = gt_xs[:, downsample_indices]
        gt_vs = gt_vs[:, downsample_indices]
        
        gt_states = (gt_xs, gt_vs)

        if self.return_downsample_indices:
            return init_state, actions, gt_states, downsample_indices
        else:
            return init_state, actions, gt_states
