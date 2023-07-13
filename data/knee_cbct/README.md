## Data Preprocessing

For our collected knee dataset, the raw data are organized as follows. FL, FR, ML, and MR represent female-left, female-right, male-left, and male-right, respectively.

```
├── ./knee_dataset
│   ├── FL/
│   │   ├── 140400.mhd
│   │   └── 140400.raw
│   ├── FR/
│   ├── ML/
│   ├── MR/
```

The first step is to resample CT scans to have consistent spacing. In this dataset, the spacing is set to (0.8, 0.8, 0.8) mm. Then, CT scans are cropped or padded to have the same resolution of 256x256x256. For details, please refer to `./data/knee_cbct/1_preprocess.py`.

```
├── ./data/knee_cbct/
│   ├── resampled/
│   │   └── FL-140400.nii.gz
│   ├── processed/
│   │   └── FL-140400.nii.gz
```

The second step is to generate simulated 2D projections. Here, we utilize `tigre` for DRR generation. Projection configurations are given in `./data/knee_cbct/config.yaml` and we can run `bash 2_project.sh` to generate projections. The quality of projections will affect the reconstruction performance. 

```
├── ./data/knee_cbct/
│   ├── projections/
│   │   ├── FL-140400/
│   │   │   ├── visualization.png
│   │   │   └── all.pickle
```

The third step is to save processed CT scans (step-1) as many sub-files to accelerate data loading and reduce memory costs. For details, please refer to `./data/knee_cbct/3_block_save.py`.

```
├── ./data/knee_cbct/
│   ├── blocks/
│   │   ├── blocks.npy
│   │   ├── FL-140400/
│   │   │   ├── block_0.npy
│   │   │   ├── block_1.npy
│   │   │   └── ...
```

The final step is to normalize the projections using a consistent scaling factor (i.e., the maximum value of all projections). Projections are normalized to [0, 1], scaled to [0, 255], and saved as `uint8` to save disk space and accelerate data loading. For details, please refer to `./data/knee_cbct/4_normalize.py`.

```
├── ./data/knee_cbct/
│   ├── projections_normalized/
│   │   └── FL-140400.pickle
```
