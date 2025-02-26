<div align="center">
<p align="center">
  <h1>[CVPR 2025] MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds</h1>
  <a href="https://arxiv.org/abs/2412.06974">Paper</a> | <a href="https://mv-dust3rp.github.io/">Website</a> | <a href="https://www.youtube.com/watch?v=LBvnuKQ8Rso">Video</a> | <a href="https://huggingface.co/Zhenggang/MV-DUSt3R/tree/main/trajectories"> Data </a> | <a href="https://huggingface.co/Zhenggang/MV-DUSt3R/tree/main/checkpoints"> Checkpoints </a>
</p>
</div>

[Zhenggang Tang](https://recordmp3.github.io), [Yuchen Fan](https://ychfan.github.io/), [Dilin Wang](https://wdilin.github.io/), [Hongyu Xu](https://hyxu2006.github.io/),[Rakesh Ranjan](https://www.linkedin.com/in/rakesh-r-3848538), [Alexander Schwing](https://www.alexander-schwing.de/), [Zhicheng Yan](https://sites.google.com/view/zhicheng-yan)

<div class="content has-text-centered"> <img src="https://github.com/MV-DUSt3Rp/MV-DUSt3Rp.github.io/blob/main/static/images/tsr_.png" class="interpolation-image"/> </div>

## TL;DR

Multi-view Pose-free RGB-only 3D reconstruction in one step.
Also supports for new view synthesis and relative pose estimation.

Please see more visual results and video on our [website](https://mv-dust3rp.github.io/)!

## Update Logs

- 2025-1-30: data generation code of ScanNet.
- 2025-1-1: A gradio demo, all checkpoints, training/evaluation code and training/evaluation trajectories of ScanNet.
- 2025-1-8: demo view selection improved, better quality for multiple rooms.

## Installation

We only test this on a linux server and CUDA=12.4

1. Clone MV-DUSt3R+

```bash
git clone https://github.com/facebookresearch/mvdust3r.git
cd mvdust3r
```

2. Install the virtual environment under anaconda.

```bash
./install.sh
```

(version of pytorch and pytorch3d should be changed if you need other CUDA version.)

3. (Optional for faster runtime) Compile the cuda kernels for RoPE (the same as [DUSt3R and Croco](https://github.com/naver/dust3r?tab=readme-ov-file#installation))

```bash
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

## Checkpoints

Please download checkpoints [here](https://huggingface.co/Zhenggang/MV-DUSt3R/tree/main/checkpoints) to the folder [checkpoints](https://github.com/facebookresearch/mvdust3r/tree/main/checkpoints) before trying demo and evaluation.

|     Name    | Description |
|-------------|-------------|
| MVD.pth | MV-DUSt3R |
| MVDp_s1.pth | MV-DUSt3R+ trained on stage 1 (8 views) |
| MVDp_s2.pth | MV-DUSt3R+ trained on stage 1 then stage 2 (mixed 4~12 views) |
|DUSt3R_ViTLarge_BaseDecoder_224_linear.pth | the pretrained [DUSt3R model](https://github.com/naver/dust3r?tab=readme-ov-file#checkpoints). Our training is finetuned upon it |

## Gradio Demo

```bash
python demo.py --weights ./checkpoints/{CHECKPOINT}
```

You will see the UI like this:

<div class="content has-text-centered"> <img src="https://github.com/facebookresearch/mvdust3r/blob/main/static/demo1.png" class="interpolation-image"/> </div>

The input can be multiple images (we do not support a single image) or a video.
You will see the pointcloud along with predicted camera poses (3DGS visualization as future work).

The `confidence threshold` controls how many low confidence points should be filtered.
The `No. of video frames` is only valid when the input is a video and controls how many frames are uniformly selected from the video for reconstruction.

Note that the demo's inference is slower than what claimed in the paper due to overheads of gradio and model loading. If you need faster runtime, please use our evaluation code.

some [tips](https://github.com/facebookresearch/mvdust3r/issues/5#issuecomment-2578380545) to improve quality especially for multiple rooms.



## Data

We use five data for training and test: [ScanNet](https://github.com/ScanNet/ScanNet), [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/), [HM3D](https://aihabitat.org/datasets/hm3d/), [Gibson](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md), [MP3D](https://niessner.github.io/Matterport/). Please go to their website to sign contract, download and extract them in the folder [data](https://github.com/facebookresearch/mvdust3r/tree/main/data). [Here](https://github.com/facebookresearch/mvdust3r/tree/main/data) are more instructions.

Currently we released the [trajectories](https://huggingface.co/Zhenggang/MV-DUSt3R/tree/main/trajectories) of ScanNet for evaluation. Please download it to the folder [trajectories](https://github.com/facebookresearch/mvdust3r/tree/main/trajectories) More trajectories for training and more data will be released later.

## Evaluation

Here we have the following scripts for evaluation on ScanNet in the folder [scripts](https://github.com/facebookresearch/mvdust3r/tree/main/scripts):


|     Name    | Description |
|-------------|-------------|
| test_mvd.sh | MV-DUSt3R |
| test_mvdp_stage1.sh | MV-DUSt3R+ trained on stage 1 (8 views) |
| test_mvdp_stage2.sh | MV-DUSt3R+ trained on stage 1 then stage 2 (mixed 4~12 views) |

They should reproduce the [paper](https://arxiv.org/pdf/2412.06974)'s result on ScanNet (Tab. 2, 3, 4, S2, S3, and S5).

## Training

We are still preparing for the releasing of trajectories of training data and code of trajectory generation. Here we also put training scripts in the folder [scripts](https://github.com/facebookresearch/mvdust3r/tree/main/scripts), which can provide more information about our training.


|     Name    | Description |
|-------------|-------------|
| train_mvd.sh | MV-DUSt3R, loaded from DUSt3R to finetune |
| train_mvdp_stage1.sh | MV-DUSt3R+ training on stage 1 (8 views), loaded from DUSt3R to finetune |
| train_mvdp_stage2.sh | MV-DUSt3R+ trained on stage 1 finetuning on stage 2 (mixed 4~12 views) |

## Citation

```bibtex
@article{tang2024mv,
  title={MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds},
  author={Tang, Zhenggang and Fan, Yuchen and Wang, Dilin and Xu, Hongyu and Ranjan, Rakesh and Schwing, Alexander and Yan, Zhicheng},
  journal={arXiv preprint arXiv:2412.06974},
  year={2024}
}
```

## License

We use [CC BY-NC 4.0](https://github.com/facebookresearch/mvdust3r/tree/main/LICENSE)

## Acknowledgement

Many thanks to:
- [DUSt3R](https://github.com/naver/dust3r) for the codebase.
