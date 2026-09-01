# Particle-Grid Neural Dynamics for Learning Deformable Object Models from RGB-D Videos

<span class="author-block">
<a target="_blank" href="https://kywind.github.io/">Kaifeng Zhang</a><sup>1</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://boey-li.github.io/">Baoyu Li</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://kkhauser.web.illinois.edu/">Kris Hauser</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://yunzhuli.github.io/">Yunzhu Li</a><sup>1</sup>
</span>

<span class="author-block"><sup>1</sup>Columbia University,</span>
<span class="author-block"><sup>2</sup>University of Illinois Urbana-Champaign</span>

[Website](https://kywind.github.io/pgnd) | [Paper](https://arxiv.org/abs/2506.15680)

<img src="imgs/teaser.png" width="100%"/>

## Updates
- **[2026/01/05]** Minor updates to the training and evaluation scripts; fixed bugs, cleaned the dataset, updated dataset links, and revised the installation guide.

- **[2025/07/07]** Updated the custom dataset processing pipeline.

- **[2025/06/18]** Initial release of the PGND codebase, including training and evaluation scripts, datasets, and pretrained checkpoints.

## Interactive Demo
We provide an [interactive demo](https://huggingface.co/spaces/kaifz/pgnd) in Huggingface Spaces.

## Installation

### Prerequisite

We recommend installing the latest version of CUDA (12.x) and PyTorch. The CUDA version used to compile PyTorch should be the same as the system's CUDA version to enable installation of the ```diff_gaussian_rasterizer``` package.

### Setup an environment

1. Prepare python environment
```
conda create -n pgnd python=3.10
conda activate pgnd
```

2. Install ```dgl``` and ```ffmpeg```:
```
conda install -c dglteam/label/th24_cu124 dgl  # although this requires cuda 12.4, it is tested that dgl can run normally with higher system and pytorch cuda versions.
conda install conda-forge::ffmpeg
```

3. Install PyTorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). Make sure that the pytorch cuda version matches the system cuda version (```nvcc --version```). The torch version itself does not need to be strictly constrained. For example:

```
# if the system cuda version is 12.4:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# if the system cuda version is 12.8:
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

4. Install other python packages:
```
pip install -r requirements.txt
```

4. Install ```diff_gaussian_rasterization```:
```
cd third-party/diff-gaussian-rasterization-w-depth
pip install --no-build-isolation -e .
```

## Data and Checkpoints
The dataset is available from [Hugging Face](https://huggingface.co/datasets/kaifz/pgnd-dataset).
Pre-trained checkpoint files can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1JfZ7NBdkZm8T0WSse0cwY2GhflMd0eHV).

We provide the full training and evaluation datasets for all six categories.
Each category can be downloaded independently. The repository's download
helper verifies the archive checksums and extracts the selected category and
shared Gaussian assets directly into the paths expected by this codebase:

```bash
hf download kaifz/pgnd-dataset download_category.py \
  --repo-type dataset \
  --local-dir .
python download_category.py box --output .
```

Run the last command once for each desired category (`box`, `bread`, `cloth`,
`paperbag`, `rope`, or `sloth`). The resulting layout is:

```
- experiments/log/data
  - 0112_box_processed
  - 0112_box2_processed
  - 0112_box3_processed
  - 1018_sloth_processed
  - box_merged
  - sloth_merged
  ...
```
For the checkpoints, the files should be unzipped and organized as the following:
```
- experiments/log
  - box
    - train
      - ckpts
        - 100000.pt
      - hydra.yaml
  - sloth
    - train
      - ckpts
        - 100000.pt
      - hydra.yaml
  ...
```
The path needs to match exactly for training and inference scripts to work. If you need to use data in a different format, you may need to directly modify the code to accomodate.

The Hugging Face release includes the additional Gaussian Splatting rendering
assets (`gripper.splat` and `table.splat`) and the gripper point-sampling asset
(`gripper_new.splat`). The download helper installs them at:

```
- experiments/log/gs/ckpts
  - gripper_new.splat
  - gripper.splat
  - table.splat
```

## Custom Dataset

For processing data from raw RGB-D recordings, the following pre-trained detection, segmentation, and tracking models are required. These can be installed by:
```
pip install iopath
pip install segment-anything
pip install --no-deps git+https://github.com/IDEA-Research/GroundingDINO
pip install --no-deps git+https://github.com/facebookresearch/sam2
pip install --no-deps git+https://github.com/facebookresearch/co-tracker
```
And the weights can be downloaded from [this link](https://drive.google.com/drive/folders/1AOlS6NafyrKyMyV962UeKbU0wxsDzqBw). Extract files and put them in the ```weights/``` folder.

The raw data should contain multi-view rgb and depth image recordings, and robot end-effector translation/rotation/gripper openness recordings. An example of raw data is available at 
[this link](https://drive.google.com/drive/folders/1Dk-TyyMUo2486zTKRqsqA93eXfz5zsMl) and could be extracted to ```experiments/log/data/cloth_test```. 

For data processing, the following command runs the data processing code and generates the processed dataset with point tracks: (if you are not using the example raw data, please specify the dataset name to solve, dataset dirs, and other arguments under ```if __name__ == '__main__'``` in ```postprocess.py``` according to your own dataset naming)
```
python experiments/real_world/postprocess.py
```

## Training
Once the datasets are prepared, we provide training scripts in the ```experiments/scripts``` folder. 
```
bash experiments/scripts/train_<material_name>.sh
```
Training could take several hours on a single GPU with memory >= 24GB. It is possible to plot training loss and visualize predictions during validation in wandb with cfg.debug=False. If there is a cuda out-of-memory error, you can consider reducing the batch size.

## Inference

### Eval
Use the following command to evaluate the trained policy on the evaluation dataset and verify its performance. 
The results will be saved in the training folder, with three metrics measured over the validation set:
MDE (Mean Distance Error), CD (Chamfer Distance), EMD (Earth Mover's Distance).

```
python experiments/train/eval.py --task <material_name> --state_only
```

### Eval with 3DGS Rendering

During evaluation, it is also possible to produce 3D Gaussian renderings and particle visualizations when the ```--state_only``` flag is removed. For the downloaded datasets, they already include reconstructed 3D Gaussians stored as ```.splat``` files (e.g. as in ```experiments/log/data/1018_sloth_processed/episode_0002/gs```). To launch the eval, run:

```
python experiments/train/eval.py --task <material_name>
```

For custom datasets, after data processing, there should be ```.../<data_dir_name>/episode_xxxx/pcd_clean/``` folders with ```.npz``` files storing the segmented point clouds. From here, we need to run the GS training script, to save the ```.splat``` files in the `.../<data_dir_name>/episode_xxxx/gs/` directories:

```
python experiments/real_world/reconstruct_gs.py --task <data_dir_name>
```

### Planning
It is possible to perform model-based planning for manipulation tasks using the learned dynamics model by running
```
python experiments/real_world/plan.py --config <model_yaml_path> --text_prompts <prompts>
```
This requires building a xArm robot setup with realsense cameras, and calibrating them by
```
python experiments/real_world/calibrate.py [--calibrate/--calibrate_bimanual]
```
The calibration board needs to be put in a specific position relative to the robot to fix the robot-to-board transformation, and the camera-to-board transformations are processed using OpenCV detection algorithms. For questions about real robot setups, please feel free to reach out to the first author of the paper. We are working to release a more detailed instruction on real robot experiments soon.


## Citation
If you find this repo useful for your research, please consider citing our paper
```
@inproceedings{zhang2025particle,
  title={Particle-Grid Neural Dynamics for Learning Deformable Object Models from RGB-D Videos},
  author={Zhang, Kaifeng and Li, Baoyu and Hauser, Kris and Li, Yunzhu},
  booktitle={Proceedings of Robotics: Science and Systems (RSS)},
  year={2025}
}
```
