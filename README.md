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

[Website](https://kywind.github.io/pgnd) |
[Paper](https://www.roboticsproceedings.org/rss21/p036.pdf)

<img src="imgs/teaser.png" width="100%"/>

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

2. Install PyTorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

3. Install other packages:
```
pip install -r requirements.txt
conda install -c dglteam/label/th24_cu124 dgl
conda install conda-forge::ffmpeg
```

4. Install diff_gaussian_rasterization:
```
cd third_party/diff-gaussian-rasterization-w-depth
pip install -e .
```

## Data and Checkpoints
The dataset and pre-trained checkpoint files could be downloaded from [this link](https://drive.google.com/drive/folders/10E0gpETbwA8d1XntIsP0fgWGHipndAe8?usp=sharing). 

For the dataset, We provide the full training and evaluation datasets for all six categories. The dataset is stored as a zip file for each category, e.g. for box, all the data are stored in data_box.zip. The files should be unzipped and organized as the following (take box and sloth as examples; suppose the data for these two categories are downloaded):
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

## Custom Dataset

For processing data from raw RGB-D recordings, the following pre-trained detection, segmentation, and tracking models are required. These can be installed by:
```
pip install iopath
pip install segment-anything
pip install --no-deps git+https://github.com/IDEA-Research/GroundingDINO
pip install --no-deps git+https://github.com/facebookresearch/sam2
pip install --no-deps git+https://github.com/facebookresearch/co-tracker
```
And the weights can be downloaded from [this link](https://drive.google.com/drive/folders/1Z70RgW7oiTIfvdk_qOOtqDGsMFQ0jimY?usp=sharing). Extract files and put them in the ```weights/``` folder.

The raw data should contain multi-view rgb and depth image recordings, and robot end-effector translation/rotation/gripper openness recordings. An example of raw data is available at 
[this link](https://drive.google.com/drive/folders/1QQ2ftIJZWFGzBeFg3bxqy6ktsUgmsHts?usp=drive_link) and could be extracted to ```experiments/log/data/cloth_test```. 

For data processing, the following command runs the data processing code and generates the processed dataset with point tracks:
```
python experiments/real_world/postprocess.py
```

## Training
Once the datasets are prepared, we provide training scripts in the ```experiments/scripts``` folder. 
```
bash experiments/scripts/train_<material_name>.sh
```
Training could take several hours on a single GPU with memory >= 24GB. It is possible to plot training loss and visualize predictions during validation in wandb with cfg.debug=False.

## Inference

### Eval
Use the following command to evaluate the trained policy on the evaluation dataset and verify its performance. 
The file can also produce 3D Gaussian renderings and particle visualizations when the ```--state_only``` flag is removed
and the source dataset contains reconstructed 3D Gaussians stored as ```.splat``` files (e.g. as in ```experiments/log/data/1018_sloth_processed/episode_0002/gs```).

```
python experiments/train/eval.py --task <material_name> --state_only
```
Note: in the current dataset, some categories still lack GS reconstructions, hence evaluation can only include state-based metrics. The code for reconstruction will be released soon.

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