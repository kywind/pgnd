from typing import Union
from pathlib import Path
import argparse
import os
import subprocess
import numpy as np
import glob
import cv2
import torch
import shutil
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import supervision as sv
import open3d as o3d
import time
import kornia
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import pgnd
from pgnd.utils import get_root, mkdir
from pgnd.ffmpeg import make_video
root: Path = get_root(__file__)
sys.path.append(str(root / "real_world"))

from utils.pcd_utils import visualize_o3d, depth2fgpcd
from utils.track_utils import sample_points_from_masks
from utils.env_utils import get_bounding_box, get_bounding_box_bimanual
from modules_teleop.perception import PerceptionModule

# cotracker
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer

# sam2
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


def match_timestamps(name: str, recording_dirs: dict, num_cams: int = 4):
    # post process the recording

    count = 0
    for recording_name, action_name_list in recording_dirs.items():
        for action_name in action_name_list:
            print(f"Processing {recording_name} {action_name}")
            
            save_dir = root / "log" / "data" / name
            mkdir(save_dir, overwrite=False, resume=True)

            episode_save_dir = save_dir / f"episode_{count:04d}"
            mkdir(episode_save_dir, overwrite=True, resume=False)

            episode_save_dir_cam_list = []
            for cam in range(num_cams):
                episode_save_dir_cam = episode_save_dir / f"camera_{cam}"
                episode_save_dir_cam_rgb = episode_save_dir_cam / "rgb"
                episode_save_dir_cam_depth = episode_save_dir_cam / "depth"
                mkdir(episode_save_dir_cam_rgb, overwrite=True, resume=False)
                mkdir(episode_save_dir_cam_depth, overwrite=True, resume=False)
                episode_save_dir_cam_list.append(episode_save_dir_cam)

            episode_save_dir_robot = episode_save_dir / "robot"
            mkdir(episode_save_dir_robot, overwrite=True, resume=False)

            count += 1

            # load the recording
            recording_dir = root / "log" / "data" / recording_name
            calibration_dir = recording_dir / "calibration"
            subprocess.run(f'cp -r {calibration_dir} {episode_save_dir}', shell=True)
            robot_dir = recording_dir / "robot"
            robot_timesteps = sorted([float(d[:-4]) for d in os.listdir(robot_dir)])
            action_dir = recording_dir / str(action_name)
            with open(action_dir / 'timestamps.txt', 'r') as f:
                action_timesteps = f.readlines()
            action_timesteps = [[float(tt) for tt in t.split()[-num_cams:]] for t in action_timesteps]

            # match timesteps
            # iterate over action timesteps
            # for each action timestep, find the closest robot timestep
            for t, action_timestep in enumerate(action_timesteps):
                master_timestep = action_timestep[0]
                timesteps = [master_timestep]
                for cam in range(1, num_cams):
                    min_dist = 999
                    min_dist_tt = -100
                    for tt in range(max(t-1, 0), min(t+1, len(action_timesteps))):
                        t_diff = abs(action_timesteps[tt][cam] - master_timestep)
                        if t_diff < min_dist:
                            min_dist = t_diff
                            min_dist_tt = tt
                    assert min_dist_tt != -100
                    timesteps.append(action_timesteps[min_dist_tt][cam])
                
                # find corresponding robot data
                min_dist = 999
                min_dist_tt = 100
                for tt in range(len(robot_timesteps)):
                    t_diff = abs(master_timestep - robot_timesteps[tt])
                    if t_diff < min_dist:
                        min_dist = t_diff
                        min_dist_tt = tt
                assert min_dist_tt != -100
                assert min_dist_tt > 0
                if min_dist_tt >= len(robot_timesteps) - 1:
                    print('[Warning] robot recording ends before camera recording')
                    break
                
                # saves things
                # save the matched timesteps
                with open(episode_save_dir / "timestamps.txt", 'a') as f:
                    f.write(' '.join([str(tt) for tt in timesteps]) + '\n')
                
                # save the matched timesteps
                for cam in range(num_cams):
                    source_dir = action_dir / f"camera_{cam}" / "rgb" / f"{t:06d}.jpg"
                    target_dir = episode_save_dir_cam_list[cam] / "rgb" / f"{t:06d}.jpg"
                    subprocess.run(f'cp {source_dir} {target_dir}', shell=True)

                    source_dir = action_dir / f"camera_{cam}" / "depth" / f"{t:06d}.png"
                    target_dir = episode_save_dir_cam_list[cam] / "depth" / f"{t:06d}.png"
                    subprocess.run(f'cp {source_dir} {target_dir}', shell=True)
                
                # interpolate robot motion using the closest robot timesteps
                if master_timestep > robot_timesteps[min_dist_tt]:
                    tt1 = min_dist_tt
                    tt2 = min_dist_tt + 1
                else:
                    tt1 = min_dist_tt - 1
                    tt2 = min_dist_tt
                weight = (master_timestep - robot_timesteps[tt1]) / (robot_timesteps[tt2] - robot_timesteps[tt1] + 1e-6)
                robot_data1 = np.loadtxt(robot_dir / f"{robot_timesteps[tt1]:.3f}.txt")
                robot_data2 = np.loadtxt(robot_dir / f"{robot_timesteps[tt2]:.3f}.txt")

                @torch.no_grad()
                def interpolate_matrices(mat1, mat2, weight):
                    mat1 = torch.tensor(mat1)
                    mat2 = torch.tensor(mat2)
                    quat1 = kornia.geometry.conversions.rotation_matrix_to_quaternion(mat1)
                    quat2 = kornia.geometry.conversions.rotation_matrix_to_quaternion(mat2)
                    quat1 = kornia.geometry.quaternion.Quaternion(quat1)
                    quat2 = kornia.geometry.quaternion.Quaternion(quat2)
                    quat = quat1.slerp(quat2, weight).data
                    mat = kornia.geometry.conversions.quaternion_to_rotation_matrix(quat)
                    mat = mat.numpy()
                    return mat

                assert robot_data1.shape[0] in [1, 5, 9], f'{robot_data1.shape[0]}'  # bi-manual (2 * (1 pos + 3 rot) + 1 gripper) or single arm (1 pos + 3 rot + 1 gripper or 1 pos)
                if robot_data1.shape[0] > 1:  # 5 or 9
                    gripper1 = robot_data1[-1]
                    robot_data1 = robot_data1[:-1]
                    robot_data1 = robot_data1.reshape(-1, 4, 3)
                    robot_data1_trans = robot_data1[:, 0]
                    robot_data1_rot = robot_data1[:, 1:]

                    gripper2 = robot_data2[-1]
                    robot_data2 = robot_data2[:-1]
                    robot_data2 = robot_data2.reshape(-1, 4, 3)
                    robot_data2_trans = robot_data2[:, 0]
                    robot_data2_rot = robot_data2[:, 1:]

                    robot_data_trans = robot_data1_trans * (1 - weight) + robot_data2_trans * weight
                    robot_data_rot = interpolate_matrices(robot_data1_rot, robot_data2_rot, weight)
                    robot_data_gripper = gripper1 * (1 - weight) + gripper2 * weight

                    robot_data = np.concatenate([robot_data_trans[:, None], robot_data_rot], axis=1)  # (-1, 4, 3)
                    robot_data = robot_data.reshape(-1, 3)
                    robot_data = np.concatenate([robot_data, robot_data_gripper.reshape(1, 3)], axis=0)
                
                else:
                    robot_data = robot_data1 * (1 - weight) + robot_data2 * weight

                np.savetxt(episode_save_dir_robot / f"{t:06d}.txt", robot_data)


def load_camera(episode_data_dir):
    intr = np.load(episode_data_dir / 'calibration' / 'intrinsics.npy').astype(np.float32)
    rvec = np.load(episode_data_dir / 'calibration' / 'rvecs.npy')
    tvec = np.load(episode_data_dir / 'calibration' / 'tvecs.npy')
    R = [cv2.Rodrigues(rvec[i])[0] for i in range(rvec.shape[0])]
    T = [tvec[i, :, 0] for i in range(tvec.shape[0])]
    extrs = np.zeros((len(R), 4, 4)).astype(np.float32)
    for i in range(len(R)):
        extrs[i, :3, :3] = R[i]
        extrs[i, :3, 3] = T[i]
        extrs[i, 3, 3] = 1
    return intr, extrs


class PostProcessor:

    def __init__(self, 
            name, 
            bimanual=False, 
            text_prompts='cloth.',
            cameras=None,
            eef_T=None
        ):

        self.name = name
        self.data_dir = root / "log" / "data" / name

        n_episodes = len(glob.glob(str(self.data_dir / "episode*")))
        self.episodes = np.arange(n_episodes)

        if cameras is None:
            n_cameras = 4
            self.cameras = np.arange(n_cameras)
        else:
            assert isinstance(cameras, list)
            n_cameras = len(cameras)
            self.cameras = cameras

        self.max_frames = 10000
        self.H, self.W = 480, 848
        if bimanual:
            self.bbox = get_bounding_box_bimanual()
        else:
            self.bbox = get_bounding_box()  # 3D bounding box of the scene

        self.text_prompts = text_prompts

        if eef_T is None:
            self.eef_global_T = np.array([0.0, 0.0, 0.0])
        else:
            self.eef_global_T = np.array(eef_T)

    def get_mask(self):  # deprecated
        perception_module = PerceptionModule(self.data_dir / f"sam1_vis", device='cuda')

        for episode_id in self.episodes:
            intrs, extrs = load_camera(self.data_dir / f"episode_{episode_id:04d}")
            for cam in self.cameras:
                episode_data_dir_cam = self.data_dir / f"episode_{episode_id:04d}" / f"camera_{cam}" 
                rgb_paths = sorted(glob.glob(str(episode_data_dir_cam / 'rgb' / '*.jpg')))
                depth_paths = sorted(glob.glob(str(episode_data_dir_cam / 'depth' / '*.png')))
                n_frames = min(len(rgb_paths), self.max_frames)
                for frame_id in range(n_frames):
                    print(f"Processing episode {episode_id} camera {cam} frame {frame_id}")
                    rgb_path = rgb_paths[frame_id]
                    rgb = cv2.imread(rgb_path)  # bgr
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    depth = cv2.imread(depth_paths[frame_id], cv2.IMREAD_UNCHANGED)
                    depth = depth.astype(np.float32) / 1000.0
                    mask = perception_module.get_mask(rgb, depth, intrs[cam], extrs[cam], obj_names=['white cotton rope'])
                    mask_path = rgb_path.replace('rgb', 'mask')
                    mask_path = mask_path.replace('jpg', 'png')
                    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                    cv2.imwrite(mask_path, mask * 255)

    def run_sam2(self):
        checkpoint = str(root.parent / "weights/sam2/sam2.1_hiera_large.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        image_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)

        model_id = "IDEA-Research/grounding-dino-tiny"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        perception_module = PerceptionModule(vis_path=self.data_dir / "perception_vis", device='cuda', load_model=False)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            save_dir = self.data_dir / "sam2_vis"
            
            for episode_id in self.episodes:
                episode_data_dir = self.data_dir / f"episode_{episode_id:04d}"
                intrs, extrs = load_camera(episode_data_dir)
                for cam in self.cameras:

                    episode_data_dir_cam = self.data_dir / f"episode_{episode_id:04d}" / f"camera_{cam}" 
                    
                    rgb_paths = sorted(glob.glob(str(episode_data_dir_cam / 'rgb' / '*.jpg')))
                    depth_paths = sorted(glob.glob(str(episode_data_dir_cam / 'depth' / '*.png')))
                    
                    n_frames = min(len(rgb_paths), self.max_frames)
                    seq_len = 720

                    for pivot_frame in range(0, n_frames, seq_len):
                        print(f"[run_sam2] Processing episode {episode_id} camera {cam} pivot frame {pivot_frame}")

                        if os.path.exists(save_dir / "mask" / f'episode_{episode_id:04d}_camera_{cam}' / f'pivot_frame_{pivot_frame:06d}'):
                            continue

                        masks = np.zeros((1, 1))
                        ann_frame = pivot_frame + 1
                        ann_frame_idx = 0
                        objects = [None, None]
                        no_objs = False
                        multi_objs = False
                        
                        while masks.sum() == 0:
                            if ann_frame == 0:
                                import ipdb; ipdb.set_trace()
                            ann_frame -= 1
                            print(f"[run_sam2] Finding a frame with mask for frame {ann_frame}")

                            save_dir_video_cam = save_dir / "video_frames" / f"episode_{episode_id:04d}_camera_{cam}"
                            save_dir_video_pivot = save_dir / "video_frames" / f"episode_{episode_id:04d}_camera_{cam}" / f"pivot_frame_{pivot_frame:06d}"
                            os.makedirs(save_dir_video_pivot, exist_ok=True)
                            for frame_id in range(ann_frame, min(n_frames, pivot_frame + seq_len)):  # save video
                                subprocess.run(f'cp {rgb_paths[frame_id]} {save_dir_video_pivot / f"{frame_id:06d}.jpg"}', shell=True)
                            rgb_paths_segment = rgb_paths[ann_frame:min(n_frames, pivot_frame + seq_len)]

                            rgb_path = rgb_paths[ann_frame]
                            image = Image.open(rgb_path)

                            # ground
                            inputs = processor(images=image, text=self.text_prompts, return_tensors="pt").to(device)
                            with torch.no_grad():
                                outputs = grounding_model(**inputs)
                            results = processor.post_process_grounded_object_detection(
                                outputs,
                                inputs.input_ids,
                                box_threshold=0.25,
                                text_threshold=0.3,
                                target_sizes=[image.size[::-1]]
                            )
                            input_boxes = results[0]["boxes"].cpu().numpy()
                            objects = results[0]["labels"]

                            if len(objects) == 0:
                                no_objs = True
                                break
                            if len(objects) > 1:
                                objects_masked = []
                                input_boxes_masked = []
                                depth = cv2.imread(depth_paths[ann_frame], cv2.IMREAD_UNCHANGED) / 1000.0
                                mask = perception_module.get_mask_raw(depth, intrs[cam], extrs[cam])
                                for i, obj in enumerate(objects):
                                    if obj == '':
                                        continue
                                    box = input_boxes[i].astype(int)
                                    mask_box = mask[box[1]:box[3], box[0]:box[2]]
                                    if mask_box.sum() > 0: # and not (mask_box.shape[0] > 200 and mask_box.shape[1] > 300):
                                        objects_masked.append(obj)
                                        input_boxes_masked.append(box)
                                objects = objects_masked
                                input_boxes = input_boxes_masked
                                if len(objects) == 0:
                                    no_objs = True
                                    break
                                elif len(objects) > 1:
                                    multi_objs = True

                            image_predictor.set_image(np.array(image.convert("RGB")))
                            masks, scores, logits = image_predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=input_boxes,
                                multimask_output=False,
                            )
                            if masks.ndim == 3:
                                masks = masks[None]
                                scores = scores[None]
                                logits = logits[None]
                            elif masks.ndim == 4:
                                assert multi_objs
                        
                        if no_objs:
                            print(episode_id, cam, 'no_objs')
                            if os.path.exists(save_dir.parent / "sam2_select_images_mask" / f'{episode_id}_{cam}_{pivot_frame}.png'):
                                img = cv2.imread(
                                    str(save_dir.parent / "sam2_select_images_mask" / f'{episode_id}_{cam}_{pivot_frame}.png'),
                                    cv2.IMREAD_UNCHANGED
                                )
                                alpha = img[:, :, 3] / 255.0
                                masks = alpha > 0

                                h_max = np.where(masks.sum(1) > 0)[0].max() + 3
                                h_min = np.where(masks.sum(1) > 0)[0].min() - 3
                                w_max = np.where(masks.sum(0) > 0)[0].max() + 3
                                w_min = np.where(masks.sum(0) > 0)[0].min() - 3

                                objects = ['paper']
                                input_boxes = np.array([[w_min, h_min, w_max, h_max]]).astype(np.float32)

                                masks = masks[None, None]
                                scores = np.ones((1, 1))
                                logits = np.ones((1, 1))

                            else:
                                os.makedirs(episode_data_dir_cam / "mask", exist_ok=True)
                                for frame_id in range(len(rgb_paths_segment)):
                                    mask = np.zeros((image.height, image.width), dtype=np.uint8)
                                    cv2.imwrite(episode_data_dir_cam / "mask" / f"{(frame_id + ann_frame):06d}.png", mask)
                                continue

                        PROMPT_TYPE_FOR_VIDEO = "box"
                        inference_state = video_predictor.init_state(video_path=str(save_dir_video_pivot))

                        if PROMPT_TYPE_FOR_VIDEO == "point":
                            # sample the positive points from mask for each objects
                            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

                            for object_id, (label, points) in enumerate(zip(objects, all_sample_points), start=1):
                                labels = np.ones((points.shape[0]), dtype=np.int32)
                                _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=ann_frame_idx,
                                    obj_id=object_id,
                                    points=points,
                                    labels=labels,
                                )
                        # Using box prompt
                        elif PROMPT_TYPE_FOR_VIDEO == "box":
                            for object_id, (label, box) in enumerate(zip(objects, input_boxes), start=1):
                                _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=ann_frame_idx,
                                    obj_id=object_id,
                                    box=box,
                                )
                        # Using mask prompt is a more straightforward way
                        elif PROMPT_TYPE_FOR_VIDEO == "mask":
                            for object_id, (label, mask) in enumerate(zip(objects, masks), start=1):
                                labels = np.ones((1), dtype=np.int32)
                                _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                                    inference_state=inference_state,
                                    frame_idx=ann_frame_idx,
                                    obj_id=object_id,
                                    mask=mask
                                )
                        else:
                            raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")
                        
                        video_segments = {}  # video_segments contains the per-frame segmentation results
                        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                            video_segments[out_frame_idx] = {
                                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }
                        
                        del inference_state
                        torch.cuda.empty_cache()
                        
                        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(objects, start=1)}

                        # dir to save intermediate results
                        save_dir_mask_cam = save_dir / "mask" / f"episode_{episode_id:04d}_camera_{cam}"
                        save_dir_mask_pivot = save_dir_mask_cam / f"pivot_frame_{pivot_frame:06d}"
                        os.makedirs(save_dir_mask_pivot, exist_ok=True)

                        # dir to save final results
                        os.makedirs(episode_data_dir_cam / "mask", exist_ok=True)

                        for idx, (frame_idx, segments) in enumerate(video_segments.items()):
                            if idx != frame_idx:
                                import ipdb; ipdb.set_trace()
                            try:
                                img = cv2.imread(os.path.join(save_dir_video_pivot, rgb_paths_segment[frame_idx]))
                            except:
                                import ipdb; ipdb.set_trace()
                            
                            object_ids = list(segments.keys())
                            masks = list(segments.values())
                            masks = np.concatenate(masks, axis=0)

                            if masks.shape[0] > 1:
                                assert multi_objs
                                masks_save = np.logical_or.reduce(masks, axis=0, keepdims=True)
                            else:
                                masks_save = masks
                            cv2.imwrite(episode_data_dir_cam / "mask" / f"{(frame_idx + ann_frame):06d}.png", masks_save[0] * 255)

                            vis = True
                            if vis:
                                detections = sv.Detections(
                                    xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                                    mask=masks, # (n, h, w)
                                    class_id=np.array(object_ids, dtype=np.int32),
                                )
                                box_annotator = sv.BoxAnnotator()
                                annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
                                label_annotator = sv.LabelAnnotator()
                                annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
                                mask_annotator = sv.MaskAnnotator()
                                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                                cv2.imwrite(save_dir_mask_pivot / f"annotated_frame_{(frame_idx + ann_frame):06d}.jpg", annotated_frame)


    def get_tracking(self, target_frames=None):
        if target_frames is not None:
            target_frames = tuple(target_frames)

        save_dir = self.data_dir / f"cotracker_vis"
        os.makedirs(save_dir, exist_ok=True)

        cotracker_predictor = CoTrackerPredictor(checkpoint=root.parent / "weights/scaled_offline.pth", v2=False, offline=True, window_len=60).to('cuda')
        visualizer = Visualizer(save_dir=save_dir / "pred_video", pad_value=120, linewidth=3)

        for episode_id in self.episodes:
            episode_data_dir = self.data_dir / f"episode_{episode_id:04d}"
            intrs, extrs = load_camera(episode_data_dir)
            for cam in self.cameras:
                start_frame = 0

                episode_data_dir_cam = self.data_dir / f"episode_{episode_id:04d}" / f"camera_{cam}" 
                rgb_paths = sorted(glob.glob(str(episode_data_dir_cam / 'rgb' / '*.jpg')))
                mask_paths = sorted(glob.glob(str(episode_data_dir_cam / 'mask' / '*.png')))
                depth_paths = sorted(glob.glob(str(episode_data_dir_cam / 'depth' / '*.png')))

                os.makedirs(episode_data_dir_cam / "vel", exist_ok=True)
                os.makedirs(episode_data_dir_cam / "depth_mask", exist_ok=True)
                
                n_frames = min(len(rgb_paths), self.max_frames)
                pivot_skip = 5
                seq_len = 15
                gap = 5

                save_dir_speed_cam = save_dir / "pred_speed" / f"episode_{episode_id:04d}_camera_{cam}"
                os.makedirs(save_dir_speed_cam, exist_ok=True)

                if target_frames is None:
                    pivot_frames = range(start_frame, n_frames - gap, pivot_skip)
                else:  # only the pivot windows covering the requested frames; each window is independent
                    pivot_frames = sorted({
                        (f // pivot_skip) * pivot_skip
                        for f in target_frames
                        if start_frame <= f < n_frames - gap
                    })

                for pivot_frame in pivot_frames:  # determine the speed for frame (pivot_frame, pivot_frame + pivot_skip)
                    print(f"[get_tracking] Processing episode {episode_id} camera {cam} pivot frame {pivot_frame}")
                    mask_pivot = cv2.imread(mask_paths[pivot_frame], cv2.IMREAD_UNCHANGED)
                    img = cv2.imread(rgb_paths[pivot_frame])  # bgr
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_masked = img.copy() * (mask_pivot[:, :, None] > 0)

                    if os.path.exists(save_dir / "temp_video_frames"):
                        shutil.rmtree(save_dir / "temp_video_frames")
                    os.makedirs(save_dir / "temp_video_frames", exist_ok=True)
                    
                    frames = []

                    # find the bounding box of the mask
                    no_mask = False
                    mask_min_h_all, mask_min_w_all, mask_max_h_all, mask_max_w_all = self.H, self.W, 0, 0
                    for frame_id in range(pivot_frame, min(pivot_frame + seq_len, n_frames)):
                        mask = cv2.imread(mask_paths[frame_id], cv2.IMREAD_UNCHANGED)
                        if mask.sum() == 0:
                            no_mask = True
                            break
                        mask_h_accum = np.cumsum(mask.sum(1), axis=0)
                        mask_w_accum = np.cumsum(mask.sum(0), axis=0)
                        mask_min_h = max(0, np.max(np.where(mask_h_accum == mask_h_accum[0])) - 5)
                        mask_min_w = max(0, np.max(np.where(mask_w_accum == mask_w_accum[0])) - 5)
                        mask_max_h = min(self.H, np.min(np.where(mask_h_accum == mask_h_accum[-1])) + 5)
                        mask_max_w = min(self.W, np.min(np.where(mask_w_accum == mask_w_accum[-1])) + 5)
                        mask_min_h_all = min(mask_min_h_all, mask_min_h)
                        mask_min_w_all = min(mask_min_w_all, mask_min_w)
                        mask_max_h_all = max(mask_max_h_all, mask_max_h)
                        mask_max_w_all = max(mask_max_w_all, mask_max_w)
                    if no_mask:
                        mask_min_h_all = 0
                        mask_max_h_all = 200
                        mask_min_w_all = 0
                        mask_max_w_all = 200
                    else:
                        center = ((mask_max_h_all + mask_min_h_all) // 2, (mask_max_w_all + mask_min_w_all) // 2)
                        max_w_h = max(mask_max_h_all - mask_min_h_all, mask_max_w_all - mask_min_w_all)
                        mask_min_h_all = max(0, center[0] - max_w_h // 2)
                        mask_max_h_all = min(self.H, center[0] + max_w_h // 2)
                        mask_min_w_all = max(0, center[1] - max_w_h // 2)
                        mask_max_w_all = min(self.W, center[1] + max_w_h // 2)
                    
                    for frame_id in range(pivot_frame, min(pivot_frame + seq_len, n_frames)):
                        mask = cv2.imread(mask_paths[frame_id], cv2.IMREAD_UNCHANGED)
                        img = cv2.imread(rgb_paths[frame_id])  # bgr
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img[mask_min_h_all:mask_max_h_all, mask_min_w_all:mask_max_w_all]
                        img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_LINEAR)
                        mask = mask[mask_min_h_all:mask_max_h_all, mask_min_w_all:mask_max_w_all]
                        mask = cv2.resize(mask, (mask.shape[1] * 4, mask.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
                        img_masked = img.copy() * (mask[:, :, None] > 0)
                        cv2.imwrite(save_dir / "temp_video_frames" / f"{frame_id:06d}.jpg", cv2.cvtColor(img_masked, cv2.COLOR_RGB2BGR))
                        frames.append(torch.tensor(img_masked))
                    
                    video = torch.stack(frames).permute(0, 3, 1, 2)[None].float().to('cuda')  # B T C H W
                    mask_pivot = mask_pivot[mask_min_h_all:mask_max_h_all, mask_min_w_all:mask_max_w_all]
                    mask_pivot = cv2.erode(mask_pivot, np.ones((3, 3), np.uint8), iterations=1)
                    pred_tracks, pred_visibility = cotracker_predictor(
                        video, 
                        segm_mask=torch.from_numpy(mask_pivot).float().to('cuda')[None, None],
                        grid_size=80,
                    ) # B T N 2,  B T N
                    
                    # smooth the tracks
                    for _ in range(3):
                        for i in range(1, pred_tracks.shape[1] - 1):
                            pred_tracks[:, i] = (2 * pred_tracks[:, i] + pred_tracks[:, i-1] + pred_tracks[:, i+1]) // 4
                    
                    vis = False
                    if vis:
                        visualizer.visualize(video, pred_tracks, pred_visibility, filename=f"episode_{episode_id:04d}_camera_{cam}_pivot_frame_{pivot_frame:06d}.mp4")

                    # transform pred tracks and pred visibility to original image size
                    pred_tracks = pred_tracks.squeeze(0).cpu().numpy()  # T N 2
                    pred_visibility = pred_visibility.squeeze(0).cpu().numpy()  # T N 1
                    pred_tracks[:, :, 0] = pred_tracks[:, :, 0] / 4 + mask_min_w_all
                    pred_tracks[:, :, 1] = pred_tracks[:, :, 1] / 4 + mask_min_h_all
                    pred_tracks = pred_tracks[:, :, ::-1].copy()
                    
                    # calculate point speed in 3D
                    for target_frame in range(pivot_frame, min(pivot_frame + pivot_skip, n_frames - gap)):
                        depth_now = cv2.imread(depth_paths[target_frame], cv2.IMREAD_UNCHANGED) / 1000.0
                        try:
                            depth_future = cv2.imread(depth_paths[target_frame + gap], cv2.IMREAD_UNCHANGED) / 1000.0
                        except:
                            import ipdb; ipdb.set_trace()
                        
                        mask_now = cv2.imread(mask_paths[target_frame], cv2.IMREAD_UNCHANGED)
                        mask_now_xy = np.where(mask_now > 0)
                        mask_now_xy = np.stack(mask_now_xy, axis=1)  # (n, 2)
                        track_now = pred_tracks[target_frame - pivot_frame]
                        vis_now = pred_visibility[target_frame - pivot_frame]

                        if len(track_now) == 0:
                            indices = np.zeros((0, 4)).astype(int)
                        else:
                            # kd tree between mask_now_xy and track_now and find top k nearest indices in track_now for each point in mask_now_xy
                            knn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree').fit(track_now)
                            _, indices = knn.kneighbors(mask_now_xy)

                        mask_future = cv2.imread(mask_paths[target_frame + gap], cv2.IMREAD_UNCHANGED)
                        mask_future_xy = np.where(mask_future > 0)
                        mask_future_xy = np.stack(mask_future_xy, axis=1)  # (n, 2)
                        track_future = pred_tracks[target_frame - pivot_frame + gap]
                        vis_future = pred_visibility[target_frame - pivot_frame + gap]

                        tracks_indices = track_now[indices]  # (n, k, 2)
                        tracks_future_indices = track_future[indices]  # (n, k, 2)
                        vis_indices = vis_now[indices]  # (n, k, 1)
                        vis_indices = np.all(vis_indices, axis=1).reshape(-1)
                        vis_future_indices = vis_future[indices]  # (n, k, 1)
                        vis_future_indices = np.all(vis_future_indices, axis=1).reshape(-1)
                        pred_mask_now_xy = np.round(tracks_indices.mean(axis=1)).astype(int)  # (n, 2)
                        pred_mask_now_xy[:, 0] = np.clip(pred_mask_now_xy[:, 0], 0, self.H - 1)
                        pred_mask_now_xy[:, 1] = np.clip(pred_mask_now_xy[:, 1], 0, self.W - 1)
                        pred_mask_future_xy = np.round(tracks_future_indices.mean(axis=1)).astype(int)  # (n, 2)  # actually we dont need to round here

                        ## extract depth and project to world coordinates
                        # get points
                        points_now = depth2fgpcd(depth_now, intrs[cam])
                        points_future = depth2fgpcd(depth_future, intrs[cam])
                        points_now = (np.linalg.inv(extrs[cam]) @ np.concatenate([points_now, np.ones((points_now.shape[0], 1)).astype(np.float32)], axis=1).T).T[:, :3]  # (N, 3)
                        points_future = (np.linalg.inv(extrs[cam]) @ np.concatenate([points_future, np.ones((points_future.shape[0], 1)).astype(np.float32)], axis=1).T).T[:, :3]  # (N, 3)
                        points_now = points_now.reshape(depth_now.shape[0], depth_now.shape[1], 3)
                        points_future = points_future.reshape(depth_future.shape[0], depth_future.shape[1], 3)

                        # mask points
                        depth_threshold = [0.0, 2.0]

                        depth_mask_now = np.logical_and((depth_now > depth_threshold[0]), (depth_now < depth_threshold[1]))  # (H, W)
                        depth_mask_now_bbox = np.logical_and(
                            np.logical_and(points_now[:, :, 0] > self.bbox[0][0], points_now[:, :, 0] < self.bbox[0][1]),
                            np.logical_and(points_now[:, :, 1] > self.bbox[1][0], points_now[:, :, 1] < self.bbox[1][1])
                        )  # does not include z axis
                        depth_mask_now_bbox = depth_mask_now_bbox.reshape(depth_now.shape[0], depth_now.shape[1])
                        depth_mask_now = np.logical_and(depth_mask_now, depth_mask_now_bbox)
                        depth_mask_now_xy = depth_mask_now[pred_mask_now_xy[:, 0], pred_mask_now_xy[:, 1]].reshape(-1)
                        depth_mask_now_xy = np.logical_and(depth_mask_now_xy > 0, np.logical_and(vis_indices, vis_future_indices))  # filter out invalid points
                        valid_idx = np.where(depth_mask_now_xy > 0)[0]
                        
                        mask_now_xy = mask_now_xy[valid_idx]
                        pred_mask_now_xy = pred_mask_now_xy[valid_idx]
                        pred_mask_future_xy = pred_mask_future_xy[valid_idx]

                        depth_mask_future = np.logical_and((depth_future > depth_threshold[0]), (depth_future < depth_threshold[1]))  # (H, W)
                        depth_mask_future_bbox = np.logical_and(
                            np.logical_and(points_future[:, :, 0] > self.bbox[0][0], points_future[:, :, 0] < self.bbox[0][1]),
                            np.logical_and(points_future[:, :, 1] > self.bbox[1][0], points_future[:, :, 1] < self.bbox[1][1])
                        )  # does not include z axis
                        depth_mask_future_bbox = depth_mask_future_bbox.reshape(depth_future.shape[0], depth_future.shape[1])
                        depth_mask_future = np.logical_and(depth_mask_future, depth_mask_future_bbox)
                        depth_mask_future_xy = depth_mask_future[mask_future_xy[:, 0], mask_future_xy[:, 1]].reshape(-1)
                        valid_idx_future = np.where(depth_mask_future_xy > 0)[0]
                        mask_future_xy_valid = mask_future_xy[valid_idx_future]

                        if valid_idx_future.shape[0] < 4 or valid_idx.shape[0] < 4:
                            print(f"Warning: not enough valid points for frame {target_frame}")
                            speed_now = np.zeros((self.H, self.W, 3))
                            speed_now_norm = np.linalg.norm(speed_now, axis=2)

                        else:
                            # proj tracks
                            k = 4
                            knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(mask_future_xy_valid)
                            _, indices = knn.kneighbors(pred_mask_future_xy)
                            proj_mask_future_xy = mask_future_xy_valid[indices]  # (n, k, 2)
                            proj_mask_future_xy = proj_mask_future_xy.reshape(-1, 2)
                            proj_mask_future_xy = np.round(proj_mask_future_xy).astype(int)
                            proj_mask_future_xy[:, 0] = np.clip(proj_mask_future_xy[:, 0], 0, self.H - 1)
                            proj_mask_future_xy[:, 1] = np.clip(proj_mask_future_xy[:, 1], 0, self.W - 1)
                            
                            points_now = points_now[pred_mask_now_xy[:, 0], pred_mask_now_xy[:, 1]].reshape(-1, 3)  # mask_now_xy or pred_mask_now_xy (i.e., track_now) ?
                            points_future = points_future[proj_mask_future_xy[:, 0], proj_mask_future_xy[:, 1]]
                            points_future = points_future.reshape(-1, k, 3).mean(axis=1)  # average the k points

                            speed = points_future - points_now  # actually velocity
                            speed /= (1. / 30. * gap)  # divide by the time interval
                            
                            speed_now = np.zeros((self.H, self.W, 3))
                            speed_now[mask_now_xy[:, 0], mask_now_xy[:, 1]] = speed
                            speed_now_norm = np.linalg.norm(speed_now, axis=2)
                            
                            outlier_mask = speed_now_norm > min(1, speed_now_norm.mean() + 50 * speed_now_norm.std())
                            outlier_xy = np.stack(np.where(outlier_mask), axis=1)  # (n, 2)
                            for xy in outlier_xy:
                                speed_now[xy[0], xy[1]] = (speed_now[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2].sum(axis=(0, 1)) - speed_now[xy[0], xy[1]]) / 8
                            speed_now_norm = np.linalg.norm(speed_now, axis=2)

                        # visualize the speed
                        viz = True
                        if viz:
                            speed_running_max = 0.5
                            _ = cv2.applyColorMap((speed_now_norm / speed_running_max * 255).astype(np.uint8), cv2.COLORMAP_JET)
                            cv2.imwrite(save_dir_speed_cam / f"{target_frame:06d}.jpg", speed_now_norm / speed_running_max * 255)

                        # save the 3d vel
                        np.savez_compressed(episode_data_dir_cam / "vel" / f"{target_frame:06d}.npz", vel=speed_now.astype(np.float16))

                        # vis depth mask
                        depth_mask_vis = np.logical_and(depth_mask_now, mask_now)
                        cv2.imwrite(episode_data_dir_cam / "depth_mask" / f"{target_frame:06d}.png", depth_mask_vis * 255)


    def get_pcd(self, frame_ids=None):
        if frame_ids is not None:
            frame_ids = tuple(frame_ids)

        for episode_id in self.episodes:
            episode_data_dir = self.data_dir / f"episode_{episode_id:04d}"
            os.makedirs(episode_data_dir / "pcd_clean", exist_ok=True)
            intrs, extrs = load_camera(episode_data_dir)

            episode_data_dir_cam_0 = self.data_dir / f"episode_{episode_id:04d}" / f"camera_{self.cameras[0]}" 
            rgb_paths = sorted(glob.glob(str(episode_data_dir_cam_0 / 'rgb' / '*.jpg')))
            mask_paths = sorted(glob.glob(str(episode_data_dir_cam_0 / 'mask' / '*.png')))
            depth_paths = sorted(glob.glob(str(episode_data_dir_cam_0 / 'depth' / '*.png')))
            vel_paths = sorted(glob.glob(str(episode_data_dir_cam_0 / "vel" / '*.npz')))
            depth_mask_paths = sorted(glob.glob(str(episode_data_dir_cam_0 / "depth_mask" / '*.png')))
            n_frames = min(len(rgb_paths), self.max_frames) - 5  # skip the last 5 frames

            selected_frame_ids = range(n_frames) if frame_ids is None else (
                frame_id for frame_id in frame_ids if 0 <= frame_id < n_frames
            )
            for frame_id in selected_frame_ids:

                print(f"[get_pcd] Processing episode {episode_id} frame {frame_id}")
                pts_list = []
                colors_list = []
                vels_list = []
                camera_indices_list = []

                for cam in self.cameras:
                    # explicit filenames: vel/depth_mask may be sparse when built via target_frames
                    episode_data_dir_cam = self.data_dir / f"episode_{episode_id:04d}" / f"camera_{cam}"
                    rgb_path = str(episode_data_dir_cam / 'rgb' / f'{frame_id:06d}.jpg')
                    mask_path = str(episode_data_dir_cam / 'depth_mask' / f'{frame_id:06d}.png')
                    depth_path = str(episode_data_dir_cam / 'depth' / f'{frame_id:06d}.png')
                    vel_path = str(episode_data_dir_cam / 'vel' / f'{frame_id:06d}.npz')

                    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    img = cv2.imread(rgb_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0
                    vel = np.load(vel_path)['vel'].astype(np.float32)

                    points = depth2fgpcd(depth, intrs[cam])
                    points = (np.linalg.inv(extrs[cam]) @ np.concatenate([points, np.ones((points.shape[0], 1)).astype(np.float32)], axis=1).T).T[:, :3]  # (N, 3)
                    points = points.reshape(depth.shape[0], depth.shape[1], 3)
                    points = points[mask > 0]

                    colors = img[mask > 0]
                    vel = vel[mask > 0]

                    assert points.shape[0] == vel.shape[0]
                    camera_indices = np.ones(points.shape[0]) * cam

                    pts_list.append(points)
                    colors_list.append(colors)
                    vels_list.append(vel)
                    camera_indices_list.append(camera_indices)
                
                pts = np.concatenate(pts_list, axis=0)
                colors = np.concatenate(colors_list, axis=0)
                vels = np.concatenate(vels_list, axis=0)
                camera_indices = np.concatenate(camera_indices_list)
                
                rm_outlier = True
                if rm_outlier:
                    camera_indices = camera_indices[:, None].repeat(9, axis=-1).reshape(pts.shape[0], 3, 3)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.colors = o3d.utility.Vector3dVector(colors / 255)
                    pcd.normals = o3d.utility.Vector3dVector(vels)  # fake normals
                    pcd.covariances = o3d.utility.Matrix3dVector(camera_indices)

                    outliers = None
                    new_outlier = None
                    rm_iter = 0
                    while new_outlier is None or len(new_outlier.points) > 0:
                        _, inlier_idx = pcd.remove_statistical_outlier(
                            nb_neighbors = 25, std_ratio = 2.0 + rm_iter * 0.5
                        )
                        new_pcd = pcd.select_by_index(inlier_idx)
                        new_outlier = pcd.select_by_index(inlier_idx, invert=True)
                        if outliers is None:
                            outliers = new_outlier
                        else:
                            outliers += new_outlier
                        pcd = new_pcd
                        rm_iter += 1
                    
                    pts = np.array(pcd.points)
                    colors = np.array(pcd.colors)
                    vels = np.array(pcd.normals)
                    camera_indices = np.array(pcd.covariances)[:, 0, 0]

                ### clean pcd

                if pts.shape[0] > 10000:
                    downsample_indices = torch.randperm(pts.shape[0])[: 10000]
                    pts = pts[downsample_indices]
                    colors = colors[downsample_indices]
                    vels = vels[downsample_indices]
                    camera_indices = camera_indices[downsample_indices]

                n_pts_orig = pts.shape[0]

                # remove outliers
                pts_z = pts.copy()
                pts_z[:, :2] = 0  # only consider z axis
                pcd_z = o3d.geometry.PointCloud()
                pcd_z.points = o3d.utility.Vector3dVector(pts_z)
                _, inlier_idx = pcd_z.remove_radius_outlier(
                    nb_points = 100, radius = 0.02
                )
                pts = pts[inlier_idx]
                colors = colors[inlier_idx]
                vels = vels[inlier_idx]
                camera_indices = camera_indices[inlier_idx]

                # remove outliers based on vel
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(vels)  # fake points
                _, inlier_idx = pcd.remove_radius_outlier(
                    nb_points = 20, radius = 0.01
                )
                pts = pts[inlier_idx]
                colors = colors[inlier_idx]
                vels = vels[inlier_idx]
                camera_indices = camera_indices[inlier_idx]

                n_pts_clean = pts.shape[0]

                knn = NearestNeighbors(n_neighbors=20, algorithm='kd_tree').fit(pts)
                _, indices = knn.kneighbors(pts)
                indices = indices[:, 1:]  # exclude the point itself
                dists = np.linalg.norm(pts[indices] - pts[:, None], axis=2)
                print(f"[clean_pcd] episode {episode_id}, frame {frame_id}, n_pts_orig: {n_pts_orig}, n_pts_clean: {n_pts_clean}, max dist: {dists.max()}, min dist: {dists.min()}, mean dist: {dists.mean()}")
                
                weights = np.exp(-dists / 0.01)
                weights = weights / weights.sum(axis=1, keepdims=True)
                vels_smooth = (weights[:, :, None] * vels[indices]).sum(axis=1)
                vels = vels_smooth

                np.savez_compressed(episode_data_dir / "pcd_clean" / f"{frame_id:06d}.npz", pts=pts, colors=colors, vels=vels, camera_indices=camera_indices)


    def vis_pcd(self):
        for episode_id in self.episodes:
            print(f"[vis_pcd] Processing episode {episode_id}")
            episode_data_dir = self.data_dir / f"episode_{episode_id:04d}"
            pcd_paths = sorted(glob.glob(str(self.data_dir / f"episode_{episode_id:04d}" / "pcd_clean" / "*.npz")))
            n_frames = min(len(pcd_paths), self.max_frames)

            vis_robot = True
            if vis_robot:
                robot_paths = sorted(glob.glob(str(self.data_dir / f"episode_{episode_id:04d}" / "robot" / "*.txt")))
                assert len(robot_paths) - 5 == n_frames

            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()

            # geometry is the point cloud used in your animaiton
            geometry = o3d.geometry.PointCloud()
            geometry.points = o3d.utility.Vector3dVector(np.load(pcd_paths[0])['pts'])
            geometry.colors = o3d.utility.Vector3dVector(np.load(pcd_paths[0])['colors'])
            visualizer.add_geometry(geometry)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            visualizer.add_geometry(axis)
            line_set = o3d.geometry.LineSet()
            visualizer.add_geometry(line_set)

            eef_global_T = self.eef_global_T
            print(f'[vis_pcd] adding one-time end effector offset {eef_global_T}')

            # add robot mesh
            if vis_robot:
                robot = np.loadtxt(robot_paths[0])
                robot = robot + eef_global_T
                robot_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                robot_sphere.compute_vertex_normals()
                robot_sphere.paint_uniform_color([0.5, 0.5, 0.5])
                robot_sphere.translate(robot[:3])
                visualizer.add_geometry(robot_sphere)
                robot_prev = robot.copy()

            # add a tabletop mesh
            mesh = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.01)
            mesh = mesh.translate([0, -0.5, 0])
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.8, 0.8, 0.8])
            visualizer.add_geometry(mesh)

            start_frame = 1200
            for i in range(start_frame, n_frames):
                # now modify the points of your geometry
                # you can use whatever method suits you best, this is just an example
                pcd = np.load(pcd_paths[i])
                points = pcd['pts']
                colors = pcd['colors']
                vels = pcd['vels']

                geometry.points = o3d.utility.Vector3dVector(points)
                geometry.colors = o3d.utility.Vector3dVector(colors)

                arrow_length = np.linalg.norm(vels, axis=1, keepdims=True)  # Length of each arrow

                # Create endpoint of each arrow by shifting each point in a given direction
                # Here we assume the direction for each arrow is a unit vector in (0, 0, 1) (upward).
                directions = vels / np.linalg.norm(vels, axis=1, keepdims=True)
                end_points = points + arrow_length * directions

                all_points = np.vstack([points, end_points])

                # Create a LineSet to represent arrows
                lines = [[i, i + len(points)] for i in range(len(points))]
                colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for arrows

                # Define the LineSet object
                line_set.points = o3d.utility.Vector3dVector(all_points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)

                # add robot
                if vis_robot:
                    robot = np.loadtxt(robot_paths[i])
                    robot = robot + eef_global_T
                    robot_sphere.translate(robot[:3] - robot_prev[:3])
                    visualizer.update_geometry(robot_sphere)
                    robot_prev = robot.copy()

                visualizer.update_geometry(geometry)
                visualizer.update_geometry(line_set)
                visualizer.poll_events()
                visualizer.update_renderer()
                print(f"[vis_pcd] Frame {i} done")

    
    def vis_traj(self):
        for episode_id in self.episodes:
            print(f"[vis_traj] Processing episode {episode_id}")
            pcd_paths = sorted(glob.glob(str(self.data_dir / f"episode_{episode_id:04d}" / "pcd_clean" / "*.npz")))
            n_frames = min(len(pcd_paths), self.max_frames)

            start_frame = 1200
            pivot_skip = 30
            seq_len = 600

            for pivot_frame in range(start_frame, n_frames, pivot_skip):
                print(f"[vis_traj] Processing episode {episode_id} pivot frame {pivot_frame}")
                pcd = np.load(pcd_paths[pivot_frame])
                points_0 = pcd['pts']
                colors_0 = pcd['colors']
                vels_0 = pcd['vels']

                points_list = [points_0]
                vels_list = [vels_0]

                gap = 2
                dt = 1. / 30 * gap
                for frame_id in range(pivot_frame + 1, min(pivot_frame + seq_len, n_frames), gap):
                    pcd = np.load(pcd_paths[frame_id])
                    points = pcd['pts']
                    vels = pcd['vels']

                    points_pred = points_list[-1] + vels_list[-1] * dt

                    # knn
                    knn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree').fit(points)
                    _, indices = knn.kneighbors(points_pred)

                    vels_next = vels[indices].mean(axis=1)

                    points_list.append(points_pred)
                    vels_list.append(vels_next)
                
                points_list = np.stack(points_list, axis=0)
                vels_list = np.stack(vels_list, axis=0)

                vis = True
                if vis:
                    visualizer = o3d.visualization.Visualizer()
                    visualizer.create_window()

                    # geometry is the point cloud used in your animaiton
                    geometry = o3d.geometry.PointCloud()
                    geometry.points = o3d.utility.Vector3dVector(points_0)
                    geometry.colors = o3d.utility.Vector3dVector(colors_0)
                    visualizer.add_geometry(geometry)
                    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                    visualizer.add_geometry(axis)

                    # add a tabletop mesh
                    mesh = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.01)
                    mesh = mesh.translate([0, -0.5, 0])
                    mesh.compute_vertex_normals()
                    mesh.paint_uniform_color([0.8, 0.8, 0.8])
                    visualizer.add_geometry(mesh)

                    for i in range(len(points_list)):
                        # now modify the points of your geometry
                        # you can use whatever method suits you best, this is just an example
                        points = points_list[i]
                        colors = colors_0
                        vels = vels_list[i]

                        geometry.points = o3d.utility.Vector3dVector(points)
                        geometry.colors = o3d.utility.Vector3dVector(colors)

                        arrow_length = np.linalg.norm(vels, axis=1, keepdims=True)  # Length of each arrow

                        # Create endpoint of each arrow by shifting each point in a given direction
                        # Here we assume the direction for each arrow is a unit vector in (0, 0, 1) (upward).
                        directions = vels / np.linalg.norm(vels, axis=1, keepdims=True)
                        end_points = points + arrow_length * directions

                        all_points = np.vstack([points, end_points])

                        # Create a LineSet to represent arrows
                        lines = [[i, i + len(points)] for i in range(len(points))]
                        colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for arrows

                        visualizer.update_geometry(geometry)
                        visualizer.poll_events()
                        visualizer.update_renderer()
                        print(f"[pred_traj] Frame {i} done")
                        time.sleep(0.1)
                
                    visualizer.destroy_window()
                    input("Press Enter to continue...")


    def get_sub_episodes(self):
        os.makedirs(self.data_dir / "sub_episodes_v", exist_ok=True)
        sub_episode_count = 0
        
        eef_global_T = self.eef_global_T
        print(f'[get_sub_episodes] adding one-time end effector offset {eef_global_T}')
        
        def load_robot(path):
            robot = np.loadtxt(path)
            if len(robot.shape) > 1:  # 5 or 9
                assert robot.shape[0] in [5, 9]  # bi-manual (2 * (1 pos + 3 rot) + 1 gripper) or single arm (1 pos + 3 rot + 1 gripper or 1 pos)
                gripper = robot[-1]
                robot = robot[:-1]
                robot = robot.reshape(-1, 4, 3)
                robot_trans = robot[:, 0]  # (n, 3)
                robot_rot = robot[:, 1:]  # (n, 3, 3)
                if robot_trans.shape[0] == 1:  # single arm
                    gripper = gripper[:1]  # (1,)
                else:  # bi-manual
                    gripper = gripper[:2]  # (2,)
            else:
                assert len(robot.shape) == 1 and robot.shape[0] == 3
                robot_trans = robot
                robot_rot = np.eye(3)
                gripper = np.array([0.0])
            robot_trans = robot_trans + eef_global_T
            gripper = np.clip(gripper / 800.0, 0, 1)  # 1: open, 0: close
            return robot_trans, robot_rot, gripper
        
        for episode_id in self.episodes:
            episode_data_dir = self.data_dir / f"episode_{episode_id:04d}"
            intrs, extrs = load_camera(episode_data_dir)

            pcd_paths = sorted(glob.glob(str(self.data_dir / f"episode_{episode_id:04d}" / "pcd_clean" / "*.npz")))
            robot_paths = sorted(glob.glob(str(self.data_dir / f"episode_{episode_id:04d}" / "robot" / "*.txt")))
            assert len(robot_paths) - 5 == len(pcd_paths)
            n_frames = min(len(pcd_paths), self.max_frames)
            pivot_skip = 60
            seq_len = 120
            start_frame = 0
            
            for pivot_frame in range(start_frame, n_frames, pivot_skip):  # determine the speed for frame (pivot_frame, pivot_frame + pivot_skip)
                
                pcd = np.load(pcd_paths[pivot_frame])
                robot_traj, robot_rot, robot_gripper = load_robot(robot_paths[pivot_frame + 0])  # no longer +5 because that makes robot misaligned with pcd
                xyz_0 = pcd['pts']
                v_0 = pcd['vels']
                cam_indices = pcd['camera_indices']  # (n,)
                xyz_list = [xyz_0]  # after downsample
                v_list = [v_0]  # after downsample
                robot_traj_list = [robot_traj]
                robot_rot_list = [robot_rot]
                robot_gripper_list = [robot_gripper]

                gap = 1  # not self.skip_frame because we want to predict velocities with frame gap 1
                dt = 1. / 30 * gap  # dataset is fixed 30 fps

                if pivot_frame + seq_len >= n_frames:
                    continue
                end_frame = pivot_frame + seq_len
                print(f"[get_sub_episodes] Processing episode {episode_id} (sub episode {sub_episode_count}) pivot frame {pivot_frame}, end frame {end_frame}")
                
                os.makedirs(self.data_dir / "sub_episodes_v" / f"episode_{sub_episode_count:04d}", exist_ok=True)

                for frame_id in range(pivot_frame + 1, end_frame, gap):
                    pcd = np.load(pcd_paths[frame_id])
                    robot_traj, robot_rot, robot_gripper = load_robot(robot_paths[frame_id + 0])  # no longer +5 because that makes robot misaligned with pcd
                    xyz = pcd['pts']
                    v = pcd['vels']
                    xyz_pred = xyz_list[-1] + v_list[-1] * dt

                    # knn
                    knn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree').fit(xyz)
                    _, indices = knn.kneighbors(xyz_pred)
                    v_next = v[indices].mean(axis=1)

                    xyz_list.append(xyz_pred)
                    v_list.append(v_next)
                    robot_traj_list.append(robot_traj)
                    robot_rot_list.append(robot_rot)
                    robot_gripper_list.append(robot_gripper)
                
                xyz_list = np.stack(xyz_list, axis=0)
                v_list = np.stack(v_list, axis=0)
                robot_traj_list = np.stack(robot_traj_list, axis=0).reshape(len(robot_traj_list), -1)  # (T, n, 3)
                robot_rot_list = np.stack(robot_rot_list, axis=0).reshape(len(robot_rot_list), -1)  # (T, n, 3, 3)
                robot_gripper_list = np.stack(robot_gripper_list, axis=0).reshape(len(robot_gripper_list), -1)  # (T, n)

                np.savez_compressed(self.data_dir / "sub_episodes_v" / f"episode_{sub_episode_count:04d}" / f"traj.npz", 
                        xyz=xyz_list, v=v_list)
                np.savetxt(self.data_dir / "sub_episodes_v" / f"episode_{sub_episode_count:04d}" / f"cam_indices.txt", cam_indices)
                np.savetxt(self.data_dir / "sub_episodes_v" / f"episode_{sub_episode_count:04d}" / f"eef_traj.txt", robot_traj_list)
                np.savetxt(self.data_dir / "sub_episodes_v" / f"episode_{sub_episode_count:04d}" / f"eef_rot.txt", robot_rot_list)
                np.savetxt(self.data_dir / "sub_episodes_v" / f"episode_{sub_episode_count:04d}" / f"eef_gripper.txt", robot_gripper_list)
                np.savetxt(self.data_dir / "sub_episodes_v" / f"episode_{sub_episode_count:04d}" / f"meta.txt", 
                        np.array([episode_id, pivot_frame, end_frame]))

                sub_episode_count += 1


if __name__ == '__main__':

    # for example, if you have experiments/log/data/cloth_test:
    name = 'cloth_test_processed'
    text_prompts = 'cloth.'
    bimanual = True
    dirs = {
        'cloth_test/recording_1': ['1734933804']  # could have multiple entries
    }
    eef_T = [0, 0, -0.01]  # end effector offset
    cameras = None

    match_timestamps(name, dirs)

    pp = PostProcessor(name, bimanual, text_prompts, cameras=cameras, eef_T=eef_T)
    pp.run_sam2()
    pp.get_tracking()
    pp.get_pcd()
    pp.get_sub_episodes()
