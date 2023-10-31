# partly adapted from https://github.com/microsoft/human-pose-estimation.pytorch

import torch
import numpy as np
import cv2
import kornia

SCALE_FACTOR = 170

def transform_anchor_points(A, *argv):
    """
    perform matrix multiplication A*anchor_point for each body part and anchor_point
    Args:
        A: torch.tensor (batch, num_parts, 3, 3) of transformation matrices
        *args: tensors with shape (batch, num_parts, num_anchors, 3)
    """
    
    num_parts = 0
    for arg in argv:
        num_parts += arg.shape[1]
    
    assert num_parts == A.shape[1], "number of matrices should match number of parts!"

    index = 0
    transformed = []
    for arg in argv:
        num_parts = arg.shape[1]
        num_anchors = arg.shape[2]
        # repeat matrix num_anchors times
        A_ = A[:, index:index+num_parts].unsqueeze(2).repeat(1, 1, num_anchors, 1, 1)
        tr = torch.matmul(A_, arg)
        transformed.append(tr)
        index += num_parts

    return transformed


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        #print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def transform_template(input, params):
    size = input.shape[2]
    # scale up translation
    params[..., -1] = params[..., -1] * size
    return kornia.geometry.warp_affine(input, params, dsize=(size, size))

"""
Flip anchors: change left/right
input: list of anchors

flip_over_middle: True -- flip over centric x = img_size / 2 
                  False -- move around indices

"""
def flip_anchors(anchors, img_size = 256, flip_over_middle = True):
    #anchors_simple = reorder_anchors(anchors, get_midpoint = True)

    # used for ground truth
    if flip_over_middle:
        mid_axis = torch.tensor(img_size / 2.)  # calculate middle axis of anchors over which anchors will be flipped
        for i in range(len(anchors)):
            anchors[i][:, :, :, 0] = 2.0 * mid_axis - anchors[i][:, :, :, 0]
        return anchors

    else: # used for predictions
        core = anchors[0] # shape: (batch, 1, 3, 3)
        double = anchors[1] # shape: (batch, 12, 2, 3)
        single = anchors[2] # shape: (batch, 5, 1, 3)

        _flip_part(double, single)
        
        return core, double, single

"""
Flip anchors by moving indices around
"""
def _flip_part(double, single):
    # :, part id, which keypoint, which coordinate
    double_prev_left = [0,2,4,6,8,10]
    double_prev_right = [1,3,5,7,9,11]
    single_prev_left = [0,2]
    single_prev_right = [1,3]

    double_temp = double[:,double_prev_left,:,:].clone()
    single_temp = single[:,single_prev_left,:,:].clone()

    double[:,double_prev_left,:,:] = double[:,double_prev_right,:,:].clone()
    double[:,double_prev_right,:,:] = double_temp

    single[:,single_prev_left,:,:] = single[:,single_prev_right,:,:].clone()
    single[:,single_prev_right,:,:] = single_temp


"""
Flip template: change left/right
"""
def flip_template(template):
    template_flipped = template.clone()
    prev_left = [1,3,5,13,7,9,11,15]
    prev_right = [2,4,6,14,8,10,12,16]
    temp = template_flipped[:,prev_left,:,:].clone()
    template_flipped[:,prev_left,:,:] = template_flipped[:,prev_right,:,:].clone()
    template_flipped[:,prev_right,:,:] = temp
    return template_flipped

"""
Orient the anchors, depending on orient output from network
"""
def orient_anchors(anchors, orient_rounded):
    anchors_flipped = flip_anchors(anchors, flip_over_middle=False)
    anchors_oriented = []

    for i in range(len(anchors_flipped)):
        anchors_oriented.append(
            orient_rounded[...,None,None,None,None].clone() * anchors_flipped[i] + (1. - orient_rounded[...,None,None,None,None].clone()) * anchors[i].clone())
    
    return anchors_oriented

def orient_template(template, orient_rounded):
    template_flipped = flip_template(template) # shape: batch_size, num_parts, img_size, img_size
    transformed_template_2_oriented = orient_rounded[...,None,None,None].clone() * template_flipped + (1. - orient_rounded[...,None,None,None].clone()) * template
    return transformed_template_2_oriented


###########################################
# GROUND TRUTH TO PREDICTION FORMAT
###########################################

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
Scale down ground truth pose
"""
def scale_down_gtpose(bbox, pose, device):

    # transform to preprocessed frame scale
    top_left = np.floor(bbox[0] / SCALE_FACTOR)
    bottom_right = np.ceil(bbox[1] / SCALE_FACTOR)
    raster_pos = ','.join([str(top_left[0]), str(top_left[1]), str(bottom_right[0]), str(bottom_right[1])])
    bbx = get_bbx(raster_pos)

    scale = ((bbx[1][1] - bbx[0][1]) / SCALE_FACTOR)
    center = np.asarray((bbx[0, 0] + (bbx[1, 0] - bbx[0, 0]) // 2, bbx[0, 1] + (bbx[1, 1] - bbx[0, 1]) // 2))
    scale = np.asarray([scale, scale])
    
    pose = np.reshape(pose, (32, 2))
    pose = transform_points(pose, center, scale).T

    pose = torch.tensor(pose).to(device)

    return pose

"""
Transform ground truth pose to match template anchor order and data structure: [core, double, single]
shape: (batch, num_parts, num_anchors, 3)
input: ground truth pose
"""
def gt_to_anchorpts(pose, bbox, device):

    pose = scale_down_gtpose(bbox, pose, device)
    core = torch.ones(1, 1, 3, 3, 1).to(device)
    double = torch.ones(1, 12, 2, 3, 1).to(device)
    single = torch.ones(1, 5, 1, 3, 1).to(device)

    core[0,0,0,:-1,0] = pose[0]
    core[0,0,1,:-1,0] = pose[16]
    core[0,0,2,:-1,0] = pose[14]

    # left hip
    double[0,0,0,:-1,0] = pose[0]
    double[0,0,1,:-1,0] = pose[1]
    # right hip
    double[0,1,0,:-1,0] = pose[0]
    double[0,1,1,:-1,0] = pose[6]
    # left thigh
    double[0,2,0,:-1,0] = pose[1]
    double[0,2,1,:-1,0] = pose[2]
    # right thigh
    double[0,3,0,:-1,0] = pose[6]
    double[0,3,1,:-1,0] = pose[7]
    # left shin
    double[0,4,0,:-1,0] = pose[2]
    double[0,4,1,:-1,0] = pose[3]
    # right shin
    double[0,5,0,:-1,0] = pose[7]
    double[0,5,1,:-1,0] = pose[8]
    # left shoulder
    double[0,6,0,:-1,0] = pose[24]
    double[0,6,1,:-1,0] = pose[25]
    # right shoulder
    double[0,7,0,:-1,0] = pose[16]
    double[0,7,1,:-1,0] = pose[17]
    # left arm
    double[0,8,0,:-1,0] = pose[25]
    double[0,8,1,:-1,0] = pose[26]
    # right arm
    double[0,9,0,:-1,0] = pose[17]
    double[0,9,1,:-1,0] = pose[18]
    # left forearm
    double[0,10,0,:-1,0] = pose[26]
    double[0,10,1,:-1,0] = pose[27]
    # right forearm
    double[0,11,0,:-1,0] = pose[18]
    double[0,11,1,:-1,0] = pose[19]

    # left foot
    single[0,0,0,:-1,0] = pose[3]
    # right foot
    single[0,1,0,:-1,0] = pose[8]
    # left hand
    single[0,2,0,:-1,0] = pose[27]
    # right hand
    single[0,3,0,:-1,0] = pose[19]
    # head
    single[0,4,0,:-1,0] = pose[14]

    return core, double, single

def get_rotation_matrix(angle_rad):
    #ang_rad = angle * torch.pi / 180.
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle_rad.shape, 2, 2)