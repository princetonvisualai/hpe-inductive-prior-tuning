# Efficient, Self-Supervised Human Pose Estimation with Inductive Prior Tuning

## Set up

Run the following commands.

```
conda create -n poseestimation python=3.7
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c anaconda notebook
conda install -c conda-forge matplotlib
pip install -r requirements.txt
```

## Dataset access
Ask authors of Human3.6M dataset for access permission.

## Data preprocessing
```
cd code/processing
python preprocessing_create_dirs.py
python h36m_processing.py
```

## Train

`python train_baseline.py -config configs/baseline.yaml`

`python train_constrained.py -config configs/constrained.yaml`

`python train_mse_flipaugment_twostepwarp.py -config configs/mse_flipaugment_twostepwarp.yaml`

`python train_mse_flipaugment.py -config configs/mse_flipaugment.yaml`

`python train_mse_flipaugment_twostepwarp.py -config configs/mse_natural_flipaugment_twostepwarp.yaml`

`python train_mse_flipaugment.py -config configs/mse_natural_flipaugment.yaml`

`python train_mse_twostepwarp.py -config configs/mse_twostepwarp.yaml`

`python train_mse.py -config configs/mse_natural.yaml`

`python train_mse.py -config configs/mse.yaml`

`python train_baseline.py -config configs/natural.yaml`

`python train_two_step_warp.py -config configs/two_step_warp.yaml`