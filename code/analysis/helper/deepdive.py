import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

import torch
import numpy as np
import matplotlib.pyplot as plt

from visuals import (plot_by_joint, set_title, annotate_plot, show)

#######################
# Constants

METRIC = 0.05
OUT_ANGLE_DEG = 20

ARM_INDICES = [9, 10, 11, 12, 13, 14]

IMG_SIZE = 256

#######################

"""
Calculate 0.05 * bbox diagonal for each frame in dataset.
input: DATASET_LENGTH
output: DATASET_LENGTH
"""
def getBool(dists, diagonals):
    return dists < (METRIC * diagonals)[...,None]

"""
Calculate euclidean distance between two points
"""
def dist(pt1, pt2):
    return torch.sqrt(torch.sum((pt1 - pt2) ** 2, dim=-1))

"""
Get indices of poses where arms are out
"""
def get_arms_out(pose_gt, angle):
    SHOULDER_R = 0
    #ELBOW_R = 1
    WRIST_R = 2
    SHOULDER_L = 3
    #ELBOW_L = 4
    WRIST_L = 5

    arms = pose_gt[:,ARM_INDICES,:]

    r_out = _is_within_threshold(arms[:,WRIST_R, :], arms[:,SHOULDER_R, :], angle)

    bool_r = (arms[:,SHOULDER_R,0] < arms[:,WRIST_R,0]) * (r_out)

    l_out = _is_within_threshold(arms[:,SHOULDER_L, :], arms[:,WRIST_L, :], angle)

    bool_l = (arms[:,WRIST_L,0] < arms[:,SHOULDER_L,0]) * l_out

    return bool_r * bool_l

"""
Get indices of poses where arms are down
"""
def get_arms_down(pose_gt, angle):
    SHOULDER_R = 0
    #ELBOW_R = 1
    WRIST_R = 2
    SHOULDER_L = 3
    #ELBOW_L = 4
    WRIST_L = 5

    arms = pose_gt[:,ARM_INDICES,:]

    r_down = _is_greaterthan_threshold(arms[:,WRIST_R, :], arms[:,SHOULDER_R, :], angle)

    # wrist is below shoulder
    bool_r = (arms[:,WRIST_R,1] > arms[:,SHOULDER_R,1]) * (r_down)

    l_down = _is_greaterthan_threshold(arms[:,SHOULDER_L, :], arms[:,WRIST_L, :], angle)
    
    # wrist is below shoulder
    bool_l = (arms[:,WRIST_L,1] > arms[:,SHOULDER_L,1]) * l_down

    return bool_r * bool_l

"""
Is angle to horizontal line within threshold?
input: [DATASET_LENGTH, 2], the one with larger x first
"""
def _is_within_threshold(joint1, joint2, angle):
    ydist = torch.absolute(joint1[:,1] - joint2[:,1])
    allowable_ydist = torch.absolute(joint1[:,0] - joint2[:,0]) * np.tan(np.deg2rad(angle))
    return ydist <= allowable_ydist

"""
Is angle to horizontal line greater than threshold?
input: [DATASET_LENGTH, 2], the one with larger x first
"""
def _is_greaterthan_threshold(joint1, joint2, angle):
    ydist = torch.absolute(joint1[:,1] - joint2[:,1])
    allowable_ydist = torch.absolute(joint1[:,0] - joint2[:,0]) * np.tan(np.deg2rad(angle))
    return ydist >= allowable_ydist

def getStats(dists, diagonals):
    # get booleans
    bool = getBool(dists, diagonals) # want DATASET_LENGTH, NUM_JOINTS

    # pdj by joint
    pdj_by_joint = torch.mean(bool, dim=0, dtype=torch.float)
    # pdj by frame
    pdj_by_frame = torch.mean(bool, dim=-1, dtype=torch.float)
    # pdj by model
    # PDJ@0.05 = Distance between predicted and true joint < 0.05 * bbox diagonal
    pdj_overall = torch.mean(pdj_by_frame)

    # normalized l2 error
    error_normalized = (dists / IMG_SIZE)
    error_by_joint= torch.mean(error_normalized, dim=0, dtype=torch.float)
    error_by_frame = torch.mean(error_normalized, dim=-1, dtype=torch.float)
    error_overall = torch.mean(error_by_frame)

    return pdj_by_joint, pdj_by_frame, pdj_overall, error_by_joint, error_by_frame, error_overall

"""
Given 2 models, print out overall pdjs, normalize l2s, and plot pgj/normalized l2s by joint
"""
def compare_models(dists1, dists2, diagonals, model1, model2, markers, sorted_indices, colors, labels, dpi):
    pdj_by_joint1, pdj_by_frame1, pdj_overall1, error_by_joint1, error_by_frame1, error_overall1 = getStats(dists1, diagonals)
    pdj_by_joint2, pdj_by_frame2, pdj_overall2, error_by_joint2, error_by_frame2, error_overall2 = getStats(dists2, diagonals)
    print(model1)
    print("\tPDJ@0.05 1 Overall: ", float(pdj_overall1 * 100))
    print(model2)
    print("\tPDJ@0.05 2 Overall: ", float(pdj_overall2 * 100))
    print(model1)
    print("\tNormalize L2 1 Overall: ", float(error_overall1 * 100))
    print(model2)
    print("\tNormalize L2 2 Overall: ", float(error_overall2 * 100))
    
    # plot PDJ by joint
    fig, ax = plt.subplots(dpi = dpi)
    plot_by_joint(ax, pdj_by_joint1, model1, markers, sorted_indices, colors[0], ylabel="PDJ@0.05")
    plot_by_joint(ax, pdj_by_joint2, model2, markers, sorted_indices, colors[1], empty=True, ylabel="PDJ@0.05")
    set_title(ax, model1 + " vs.\n" + model2)
    stacked = torch.stack([pdj_by_joint1, pdj_by_joint2])
    annotate_plot(ax, stacked, labels, sorted_indices)
    show()

    # plot normalized error
    fig, ax = plt.subplots(dpi = dpi)
    plot_by_joint(ax, error_by_joint1, model1, markers, sorted_indices, colors[0], usePDJ=False, ylabel="Normalized L2 Error [0,1]")
    plot_by_joint(ax, error_by_joint2, model2, markers, sorted_indices, colors[1], empty=True, usePDJ=False, ylabel="Normalized L2 Error [0,1]")
    set_title(ax, model1 + " vs.\n" + model2)
    stacked = torch.stack([error_by_joint1, error_by_joint1])
    annotate_plot(ax, stacked, labels, sorted_indices)

def plot_by_joint_dist(ax, dists, diagonals, model, markers, sorted_indices, color, ylabel, usePDJ = True, empty=False, linestyle="-"):
    pdj_by_joint, pdj_by_frame, pdj_overall, error_by_joint, error_by_frame, error_overall = getStats(dists, diagonals)
    
    if usePDJ:
        plot_by_joint(ax, pdj_by_joint, model, markers, sorted_indices, color, ylabel, empty=empty, linestyle=linestyle, usePDJ = usePDJ)
        return pdj_by_joint
    else:
        plot_by_joint(ax, error_by_joint, model, markers, sorted_indices, color, ylabel, empty=empty, linestyle=linestyle, usePDJ = usePDJ)
        return error_by_joint
    
    
