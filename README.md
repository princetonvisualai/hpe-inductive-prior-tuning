# Efficient, Self-Supervised Human Pose Estimation with Inductive Prior Tuning
[Paper](https://openaccess.thecvf.com/content/ICCV2023W/ROAD++/papers/Yoo_Efficient_Self-Supervised_Human_Pose_Estimation_with_Inductive_Prior_Tuning_ICCVW_2023_paper.pdf)

<img src="sample/1.png" width="60%">

## Set up

1. In `configs/variables.py`, fill out `TO DO` elements.

2. Download Human3.6m dataset. See `Dataset access` section for details.

3. In `code/preprocessing/create_data_dict.py` and `code/preprocessing/get_frames.py` update the indexing to match the file structure of the downloaded video files (see TO DO instructions in file for detail).

4. Set up environment:

```
conda create -n poseestimation python=3.7
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c anaconda notebook
conda install -c conda-forge matplotlib
pip install -r requirements.txt
```

## Dataset access

Request authors of Human3.6M dataset for access to dataset.

Once access is granted, the data should be structures as follows in the `data/` folder.

```
data
    -- S1
        -- videos
            --mp4 files
    -- S5
        -- videos
    -- S6
        -- videos
    ...
```

## Data preprocessing

Once the data has been downloaded and programming environment is set up, preprocess the data.

```
cd code/preprocessing
python get_frames.py
python create_data_dict.py
python collect_frames_create_dirs.py
python collect_frames.py
python create_preprocessed_to_orig_mapping.py
```

## Train

First, change directory to `code/train`: `cd code/train`

Baseline: `python train_baseline.py -config ../../configs/baseline.yaml`

+ MSE: `python train_mse.py -config ../../configs/mse.yaml`

New template: `python train_baseline.py -config ../../configs/natural.yaml`

+ MSE, new template: `python train_mse.py -config ../../configs/mse_natural.yaml`

+ MSE, new template, flip augment: `python train_mse_flipaugment.py -config ../../configs/mse_natural_flipaugment.yaml`

+ MSE, new template, flip augment, coarse-to-fine: `python train_mse_flipaugment_twostepwarp.py -config ../../configs/mse_natural_flipaugment_twostepwarp.yaml`

Constrained: `python train_constrained.py -config ../../configs/constrained.yaml`

## Citation

```
@InProceedings{Yoo_2023_ICCV,
    author    = {Yoo, Nobline and Russakovsky, Olga},
    title     = {Efficient, Self-Supervised Human Pose Estimation with Inductive Prior Tuning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {3271-3280}
}
```