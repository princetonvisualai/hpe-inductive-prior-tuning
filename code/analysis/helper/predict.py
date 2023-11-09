import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

import numpy as np
import torch
import os

from analysis.helper.transforms import get_affine_transform
from utils.transforms import flip_anchors
import configs.variables as variables

########################
# CONSTANTS

NUM_JOINTS = 15
SCALE_FACTOR = 170
TEMPLATES_DIR = variables.TEMPLATES_DIR

########################

"""
Retrieve bounding box coordinates from string
"""
def get_bbx(mystr):
    bbx = np.asarray([int(float(s)) * SCALE_FACTOR for s in mystr.split(',')]).reshape((2, 2))

    return bbx

"""
Transform bbox and pose coordinates from original scale to smaller scale
"""
def transform_points(points, center, scale):
    """
    perform matrix multiplication A*anchor_point for each body part and anchor_point
    Args:
        A: torch.tensor (batch, num_parts, 3, 3) of transformation matrices
        *args: tensors with shape (batch, num_parts, num_anchors, 3)
    """

    A = get_affine_transform(center=center, scale=scale, rot=0, output_size=np.array((256, 256)))
    
    num_anchors = points.shape[0]
    
    points = np.concatenate((points, np.ones((num_anchors, 1))), axis = 1).T
    #print(np.concatenate((points, np.)))
    # repeat matrix num_anchors times
    #A_ = np.tile(np.expand_dims(A, 0), (2,1,1))
    tr = A @ points

    return tr
    
"""
Calculate midpoint of two points
"""
def midpoint(pt1, pt2):
    return (pt1 + pt2) / 2

"""
Reorder estimated joints to match order of ground truth joint locations
"""
def reorder_anchors(anchors, get_midpoint = False):
    
    anchors_cat = torch.cat([anchors[0].view(-1, 3, 1), anchors[1].view(-1, 3, 1), anchors[2].view(-1, 3, 1)])[:,0:2,0].detach().cpu().numpy()

    if get_midpoint == False:  
        core_hip = anchors_cat[0]
        core_shoulder = anchors_cat[1]
        core_neck = anchors_cat[2]
    else:
        core_hip = (anchors_cat[0] + anchors_cat[3] + anchors_cat[5]) / 3
        core_shoulder = (anchors_cat[1] + anchors_cat[15] + anchors_cat[17]) / 3
        core_neck = midpoint(anchors_cat[2], anchors_cat[31])
    
    anchors_reordered = np.array([core_hip, 
                                   midpoint(anchors_cat[6], anchors_cat[6]),
                                  midpoint(anchors_cat[10], anchors_cat[13]),
                                  midpoint(anchors_cat[14], anchors_cat[28]),
                                   midpoint(anchors_cat[4], anchors_cat[7]),
                                   midpoint(anchors_cat[8], anchors_cat[11]),
                                   midpoint(anchors_cat[12], anchors_cat[27]),
                                  core_shoulder,
                                   core_neck,
                                   midpoint(anchors_cat[18], anchors_cat[21]),
                                  midpoint(anchors_cat[22], anchors_cat[25]),
                                  midpoint(anchors_cat[26], anchors_cat[30]),
                                   midpoint(anchors_cat[16], anchors_cat[19]),
                                   midpoint(anchors_cat[20], anchors_cat[23]),
                                   midpoint(anchors_cat[24], anchors_cat[29])])
    
    return anchors_reordered


"""
Scale down pose from original scale to 256 x 256 scale
input: bbox and pose
"""
def get_pose(bbox, pose):
    # transform to preprocessed frame scale
    top_left = np.floor(bbox[0] / SCALE_FACTOR)
    bottom_right = np.ceil(bbox[1] / SCALE_FACTOR)
    raster_pos = ','.join([str(top_left[0]), str(top_left[1]), str(bottom_right[0]), str(bottom_right[1])])
    bbx = get_bbx(raster_pos)

    scale = ((bbx[1][1] - bbx[0][1]) / SCALE_FACTOR)
    center = np.asarray((bbx[0, 0] + (bbx[1, 0] - bbx[0, 0]) // 2, bbx[0, 1] + (bbx[1, 1] - bbx[0, 1]) // 2))
    scale = np.asarray([scale, scale])
    
    bbox = transform_points(bbox, center, scale).T

    pose = np.reshape(pose, (32, 2))
    pose = transform_points(pose, center, scale).T
    
    # reorder all anchors
    pose = pose[[0, 6, 7, 8, 1, 2, 3, 13, 14, 17, 18, 19, 25, 26, 27]]
    
    return bbox, pose

"""
Adjust train configurations to match test configurations
"""
def adjust_configs(config):
    config['batch_size'] = 1
    config['template_path'] = TEMPLATES_DIR + config['template_path']
    config['anchor_pts_path'] = TEMPLATES_DIR + config['anchor_pts_path']
    
"""
Return file name of most recent checkpoint
input: path name of folder containing checkpoints
"""
def most_recent_checkpoint(checkpointfolderpath):
    checkpoints = os.listdir(checkpointfolderpath)
    indices = np.array([int(''.join(filter(str.isdigit, x))) for x in checkpoints])
    
    return checkpoints[np.argmax(indices)]

"""
Sort checkpoints from oldest to most recent and return sorted list
input: path name of folder containing checkpoints
"""
def sort_checkpoints(checkpointfolderpath):
    checkpoints = np.array(os.listdir(checkpointfolderpath))
    indices = np.array([int(''.join(filter(str.isdigit, x))) for x in checkpoints])
    
    return checkpoints[np.argsort(indices)]

"""
Load predictor with specified checkpoint
"""
def load_checkpoint(predictor, checkpoint_path):
    predictor.regressor.load_state_dict(torch.load(checkpoint_path)['regressor'])
    predictor.translator.load_state_dict(torch.load(checkpoint_path)['translator'])
    
"""
Predict joints
"""
def predict_joints(predictor, frame1, frame2, get_midpoint = False):
    d = predictor.train_step(frame1, frame2, predict=True)
    anchors = d['transformed_anchors']
    anchors_reordered = reorder_anchors(anchors, get_midpoint = get_midpoint)
    
    return anchors_reordered

"""
Predict joints, reconstructed, transformed template
"""
def predict(predictor, frame1, frame2, flip = False, get_midpoint = False):
    d = predictor.train_step(frame1, frame2, return_imgs=True, predict=True)
    anchors = d['transformed_anchors']

    if flip:
        anchors = flip_anchors(anchors, flip_over_middle=False)

    anchors_reordered = reorder_anchors(anchors, get_midpoint = get_midpoint)
    
    return anchors_reordered, d['reconstructed'], d['transformed_template']