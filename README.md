# DIF-Net
Yiqun Lin, Zhongjin Luo, Wei Zhao, Xiaomeng Li, "Learning Deep Intensity Field for Extremely Sparse-View CBCT Reconstruction," MICCAI 2023. [[paper]](https://arxiv.org/abs/2303.06681)

## 0. Citation

```
@article{lin2023learning,
  title={Learning Deep Intensity Field for Extremely Sparse-View CBCT Reconstruction},
  author={Lin, Yiqun and Luo, Zhongjin and Zhao, Wei and Li, Xiaomeng},
  journal={arXiv preprint arXiv:2303.06681},
  year={2023}
}
```

## 1. Installation

```
torch 1.8.0
numpy, opencv-python, SimpleITK
```

## 2. Data Preparation

Please follow the scripts (4 steps) given in `./data/knee_cbct/*.npy` to conduct preprocessing. For detailed instructions, please refer to `./data/knee_cbct/README.md`. The processed data will be organized as follows.

```
├── ./data/knee_cbct/
│   ├── config.yaml
│   ├── info.json
│   ├── processed/
│   │   └── FL-140400.nii.gz
│   ├── blocks/
│   │   ├── blocks.npy
│   │   ├── FL-140400/
│   │   │   ├── block_0.npy
│   │   │   ├── block_1.npy
│   │   │   └── ...
│   ├── projections_normalized/
│   │   └── FL-140400.pickle
```

## 3. Training and Testing

Follow the scripts given in `./scripts/*.sh` to conduct training and testing.

## License

This repository is released under MIT License (see LICENSE file for details).
