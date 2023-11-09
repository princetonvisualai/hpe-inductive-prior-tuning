import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec

import numpy as np

import torch
import torchvision.transforms as T

plt.rcParams["savefig.bbox"] = "tight"

########################
# CONSTANTS

NUM_JOINTS = 15

INVERSE1 = T.Normalize(mean=[0, 0, 0],
                                std=[1/0.229, 1/0.224, 1/0.225])

INVERSE2 = T.Normalize(mean=[-0.485, -0.456, -0.406],
                                 std=[1, 1, 1])

INVERSE = T.Compose([INVERSE1, INVERSE2])

MARKERSIZE = 25
LINEWIDTH = 0.4

COLORS = ["#33cccc", "#ff5050", "#9900cc", "#009900", "#ff00ff", "#0080ff", "#333399", "#ff8000",
            "#00ff00", "#ff00ff", "#ccccff", "#ffff00", "#ff6699", "#ccffcc", "#008080"]
########################
"""
Plot PDJ by joint
inputs:
    ax
    pdj_by_joint1/2: pdjs by joint
    model1/2: label for each model
    labels: labels for each joint
    markers: marker style for each joint
    sorted_indices: indices by which to sort the joints
"""
def plot_by_joint(ax, pdj_by_joint, model, markers, sorted_indices, color, ylabel, empty=False, linestyle='-', usePDJ = True):
    pdj_by_joint = pdj_by_joint.cpu()
    
    """
    # then sort the joints by model 1 predictions
    if pdj_overall1 > pdj_overall2:
        sorted_indices = torch.argsort(-pdj_by_joint1)
    else: # then sort the joints by model 2 predictions
        sorted_indices = torch.argsort(-pdj_by_joint2)
    """
    pdj_by_joint = pdj_by_joint[sorted_indices]
    markers = markers[sorted_indices]
    
    x_marks = np.array(list(range(NUM_JOINTS)))
    ax.plot(x_marks, pdj_by_joint, c = color, linestyle = linestyle, linewidth = LINEWIDTH, label=model, zorder=0)

    if not empty:
        for j in range(NUM_JOINTS):
            ax.scatter(j, pdj_by_joint[j], c = color, marker=markers[j], s=MARKERSIZE, linewidth=LINEWIDTH, linestyle=linestyle)
    else:
        for j in range(NUM_JOINTS):
            ax.scatter(j, pdj_by_joint[j], marker=markers[j], c = "white", s=MARKERSIZE, linewidth=LINEWIDTH, edgecolors=color, linestyle=linestyle)
    
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
    if usePDJ:
        ax.set_ylim(-0.1, 1.15)
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    else:
        """
        # l1msssim_naturalg1_flipaugment_twostepwarp
        ax.set_ylim(-0.1, 3)
        ax.yaxis.set_ticks(np.arange(0, 3, 0.5))
        """
        
        #ax.set_ylim(0, 0.3)
        #ax.yaxis.set_ticks(np.arange(0, 0.35, 0.05))
    ax.legend(fontsize=7, framealpha=0.2, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel(ylabel)

def set_title(ax, title):
    ax.set_title(title)

# set style for the axes
def set_axis_style(ax, labels, fontsize=None):
    if fontsize is not None:
        ax.set_yticks(np.arange(1, len(labels) + 1), labels=labels, fontsize=fontsize)
    else:
        ax.set_yticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_ylim(0.25, len(labels) + 0.75)

def annotate_plot(ax, stacked, labels, sorted_indices):
    x_marks = np.array(list(range(NUM_JOINTS)))
    stacked = stacked[:,sorted_indices]
    labels = labels[sorted_indices]
    fontsize = 5

    x_marks = x_marks - 0.12
    y_berth = 0.02

    maxs = torch.max(stacked, dim=0)[0]
    for i in range(len(labels)):
        ax.annotate(labels[i], (x_marks[i], maxs[i] + y_berth), fontsize = fontsize, rotation=45)

############################################################################################################

"""
Visualize frame
    input: frame1, frame2, reconstructed
"""
def showframe(frame, inverse = True):
    fig, ax = plt.subplots()
    
    if inverse:
        frame = INVERSE(frame.squeeze())[[2,1,0],:,:]
    frame_cpu = frame.squeeze().cpu()
    frame_np = frame_cpu.detach().numpy()
    ax.imshow(frame_np.transpose((1, 2, 0)))
    
    return fig, ax

"""
Visualize the joints
    input: joints shape: [NUM_JOINTS, 2]
"""
def showjoints(joints, ax, colors=COLORS, constant = True, markersize = 10):
    if constant:
        for i in range(joints.shape[0]):
            ax.plot(joints[i,0], joints[i,1], color='red', marker='.', markeredgecolor = "none", markersize=markersize)
    else:
        for i in range(joints.shape[0]):
            ax.plot(joints[i,0], joints[i,1], color=COLORS[i], marker='.',  markeredgecolor = "none", markersize=markersize)

"""
Visualize the bbox
"""
def showbbox(bbox, ax):
    ax.add_patch(Rectangle([bbox[0,0], bbox[0,1]], 
                           bbox[1,0] - bbox[0,0], 
                          bbox[1,1] - bbox[0,1],
                          edgecolor='red',
                        facecolor='none'))
    
"""
Visualize the bbox
input: [NUM_JOINTS, 2]
"""
def showlines(joints, ax, linewidth = 2):
    linewidth = linewidth
    
    rightleg = np.array([joints[0], joints[1], joints[2], joints[3]])
    ax.plot(rightleg[:,0], rightleg[:,1], color='cyan', linewidth=linewidth)
    
    leftleg = np.array([joints[0], joints[4], joints[5], joints[6]])
    ax.plot(leftleg[:,0], leftleg[:,1], color='cyan', linewidth=linewidth)
    
    core = np.array([joints[0], joints[7], joints[8]])
    ax.plot(core[:,0], core[:,1], color='cyan', linewidth=linewidth)
    
    rightarm = np.array([joints[7], joints[9], joints[10], joints[11]])
    ax.plot(rightarm[:,0], rightarm[:,1], color='cyan', linewidth=linewidth)
    
    leftarm = np.array([joints[7], joints[12], joints[13], joints[14]])
    ax.plot(leftarm[:,0], leftarm[:,1], color='cyan', linewidth=linewidth)

"""
Render visualization
"""
def show():
    plt.show()

"""
Visualize frame, bbox, pose
"""
def visualize(frame, bbox, pose, show_bbox = False, show_lines = True, show_joints = True, constant = False):
    fig, ax = showframe(frame)
    
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if show_bbox:
        showbbox(bbox, ax)
    if show_lines:
        showlines(pose, ax)
    if show_joints:
        showjoints(pose, ax, colors=None, constant=constant)
    show()

def show_grid(grid, dpi, inverse=True, constant=True, show_lines = True, show_joints = True):
    nrow = 1
    ncol = len(grid)

    fig = plt.figure(figsize=(ncol+1, nrow + 1), dpi=dpi) 

    gs = gridspec.GridSpec(nrow, ncol,
            wspace=0.0, hspace=0.0, 
            top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

    for i in range(len(grid)):
        reconstructed, anchors = grid[i]

        if inverse:
            img = INVERSE(reconstructed.squeeze())[[2,1,0],:,:]
        img_cpu = img.squeeze().cpu()
        img_np = img_cpu.detach().numpy()
        
        ax= plt.subplot(gs[0,i])

        ax.imshow(img_np.transpose((1, 2, 0)))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if show_lines:
            showlines(anchors, ax, linewidth=0.5)
        if show_joints:
            showjoints(anchors, ax, colors=None, constant=constant, markersize=2)
        
    show()

def prep_heatmap(transformed_template):
    transformed_template = transformed_template.squeeze().cpu()
    transformed_template = transformed_template.detach().numpy()
    transformed_template = np.sum(transformed_template, 0)
    return transformed_template

def show_template(transformed_template):
    fig, ax = plt.subplots()
    transformed_template = prep_heatmap(transformed_template)
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    ax.imshow(transformed_template)
    show()
