import torch.nn as nn
import torch

l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()

################################################

l1_loss_per = nn.L1Loss(reduction='none')
l2_loss_per = nn.MSELoss(reduction='none')

def _l1_loss(x1, x2):
    return l1_loss(x1, x2)

def _l2_loss(x1, x2):
    return l2_loss(x1, x2)

def _l1_loss_per(x1, x2):
    return l1_loss_per(x1, x2)

def _l2_loss_per(x1, x2):
    return l2_loss_per(x1, x2)

# the original computer_anchor_loss
def compute_anchor_loss(core, double, single, size):
    """
    compute mean distance between pairs of transformed anchor_points,
    change this according to your connectivity constraints
    """
    loss = 0
    # normalize to range 0, 1
    core = core/size # core
    single = single/size # 'left foot', 'right foot', 'left hand', 'right hand', 'head'
    double = double/size # hip, thigh, shin, shoulder, arm, forearm

    # loss between core and hips and shoulders
    indices1 = [0, 0, 1, 1] # core indices
    # hips and shoulders
    indices2 = [0, 1, 6, 7] # double indices

    for index1, index2 in zip(indices1, indices2):

        loss += l2_loss(core[:, 0, index1], double[:, index2, 0])
    # head and core
    loss += l2_loss(core[:, 0, -1], single[:, -1, 0])

    # hips to thighs to shins, shoulders to arms to forearms
    indices3 = [0, 1, 2, 3, 6, 7, 8, 9]
    indices4 = [2, 3, 4, 5, 8, 9, 10, 11]

    for index3, index4 in zip(indices3, indices4):
        loss += l2_loss(double[:, index3, 1], double[:, index4, 0])

    #  shin to feet, forarms to hands
    indices5 = [4, 5, 10, 11]
    indices6 = [0, 1, 2, 3]

    for index5, index6 in zip(indices5, indices6):

        loss += l2_loss(double[:, index5, 1], single[:, index6, 0])

    return loss

threshold = nn.Threshold(1, 0)

def compute_boundary_loss(core, double, single, img_size):
    """
    compute boundary loss, boundaries are 0 and 1
    loss = x if x smaller or greater than 0, 1
    0 otherwise
    """
    core = core.view(core.shape[0], -1, core.shape[3])
    single = single.view(single.shape[0], -1, single.shape[3])
    double = double.view(double.shape[0], -1, double.shape[3])

    comb = torch.cat([core, double, single], dim=1)

    # normalize to range -1  to 1
    comb = (comb / img_size) * 2 - 1

    return threshold(torch.abs(comb)).sum(1).mean()

