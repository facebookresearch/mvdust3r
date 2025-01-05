<div align="center">
<p align="center">
  <h1>MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds</h1>
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

- 2025-1-1: A gradio demo, all checkpoints, training/evaluation code and training/evaluation trajectories of ScanNet.

## Installation

We only test this on a linux server and CUDA=12.4

```bash
./install.sh
```

version of pytorch and pytorch3d should be changed if you need other CUDA version.

## Checkpoints

Please download checkpoints [here](https://huggingface.co/Zhenggang/MV-DUSt3R/tree/main/checkpoints) to the folder [checkpoints](https://github.com/facebookresearch/facebookresearch/mvdust3r/checkpoints) before trying demo and evaluation.

Here we have the following checkpoints: `MVD.pth`(MV-DUSt3R), `MVDp_s1.pth`(MV-DUSt3R+, stage 1) `MVDp_s2.pth`(MV-DUSt3R+, stage 2)
`DUSt3R_ViTLarge_BaseDecoder_224_linear.pth` (the pretrained DUSt3R model. Our training is finetuned upon it)

## Gradio Demo

```bash
python demo.py --weights ./checkpoints/{CHECKPOINT}
```

## Data

We use five data for training and test: ScanNet, ScanNet++, HM3D, Gibson, MP3D. Please go to their website to sign contract, download and extract them in the folder [data](https://github.com/facebookresearch/facebookresearch/mvdust3r/data).

Currently we released the [trajectories](https://huggingface.co/Zhenggang/MV-DUSt3R/tree/main/trajectories) of ScanNet for evaluation. Please download it to the folder [trajectories](https://github.com/facebookresearch/facebookresearch/mvdust3r/trajectories) More trajectories for training and more data will be released later.

## Evaluation

Here we have the following scripts for evaluation on ScanNet in the folder [scripts](https://github.com/facebookresearch/facebookresearch/mvdust3r/scripts): `test_mvd.sh` `test_mvdp_stage1.sh` `test_mvdp_stage2.sh`.They should reproduce the paper's result on ScanNet.

## Training

We are still preparing for the releasing of trajectories of training data and code of trajectory generation. Here we also put training scripts in the folder [scripts](https://github.com/facebookresearch/facebookresearch/mvdust3r/scripts), which can provide more information about our training.

## Citation

```bibtex
@article{tang2024mv,
  title={MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds},
  author={Tang, Zhenggang and Fan, Yuchen and Wang, Dilin and Xu, Hongyu and Ranjan, Rakesh and Schwing, Alexander and Yan, Zhicheng},
  journal={arXiv preprint arXiv:2412.06974},
  year={2024}
}
```

## Acknowledgement

Many thanks to:
- [DUSt3R](https://github.com/naver/dust3r) for the codebase.
- [MMAudio](https://github.com/hkchengrex/MMAudio/tree/main) for the README structure.
