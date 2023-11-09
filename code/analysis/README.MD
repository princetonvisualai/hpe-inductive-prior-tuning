First, `cd` into the `analysis` folder.

## Step 1. Collect pose and person bbox diagonal information

`python collect_pose_diagonals.py`

## Step 2. Collect pose estimation errors

`python analysis_baseline_pretrained.py`

`python analysis.py`

`python analysis_constrained.py`

## Step 3. Collect body part length proportions of estimates

`python calc_variation.py`

`python calc_variation_constrained.py`

## Step 4. Visualize results in `qualitative.ipynb` and `quantitative.ipynb`.