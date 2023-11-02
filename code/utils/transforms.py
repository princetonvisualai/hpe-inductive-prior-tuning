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