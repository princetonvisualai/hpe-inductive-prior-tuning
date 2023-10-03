import os
import torch
from argparse import ArgumentParser
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.core.utils.helper import load_config, show_images
from src.core.model_mse_twostepwarp_orient_supervisioninteractive import Model
from src.core.utils.dataset_evensmaller_flipaugment_idxs import ImageDataset
from src.core.utils.transforms import gt_to_anchorpts, flip_anchors
import json_tricks as json

parser = ArgumentParser()

parser.add_argument('-config', help='path to config file')

args = parser.parse_args()

# loading config from yaml file
cfg = load_config(args.config)
device = cfg['device']
if device == 'cpu':
    device = torch.device('cpu')
elif device == 'gpu':
    os.environ['CUDA_AVAILABLE_DEVICES'] = cfg['gpu_num']
    device = torch.device('cuda:0')

#####################################################
core_gt = None
double_gt = None
single_gt = None
supervision_frame1 = None
supervision_frame2 = None

#####################################################
# MAJOR PARAMETERS
MILESTONE = 30
#####################################################

TOT_SUPERVISION_EXAMPLES = 2 * len(cfg['supervision_idxs'])
#####################################################
# define model
model = Model(cfg=cfg, device=device, num_examples_supervised = TOT_SUPERVISION_EXAMPLES, num_supervised_to_pick = 0)
# preprocessing of input images
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# inverse
inverse1 = transforms.Normalize(mean=[0, 0, 0],
                                std=[1/0.229, 1/0.224, 1/0.225])

inverse2 = transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                 std=[1, 1, 1])

inverse = transforms.Compose([inverse2, inverse1])

# define dataset and dataloader
print("Getting mapping")
PREPROCESSED_TO_ORIG_ALL = "/scratch/network/nobliney/project/src/core/analysis/preprocessed_to_orig_all.json"
# Get mapping from preprocessed frame path to corresponding bbox and pose
with open(PREPROCESSED_TO_ORIG_ALL, "r") as f:
    preprocessed_to_orig_all = json.load(f)
print("Finished getting mapping")
dataset = ImageDataset(transform=normalize, mapping=preprocessed_to_orig_all)

dataloader = DataLoader(dataset=dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                        drop_last=True)

num_epochs = cfg['num_epochs']
log_dir = cfg['log_dir']
writer = SummaryWriter(log_dir=log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
check_dir = cfg['checkpoint_dir']
if not os.path.exists(check_dir):
    os.makedirs(check_dir)

return_imgs = False
# training loop
i = 0

def get_gts():

    indices = cfg['supervision_idxs']

    num_examples = len(indices)
    # Augment with flipped versions
    for j in range(num_examples):
        indices.append(dataset.__len__() // 2 + indices[j])

    supervision_frame1 = []
    supervision_frame2 = []
    core_gt = []
    double_gt = []
    single_gt = []
    for j in indices:
        frame1, frame2, bbox1, pose1 = dataset.getitem_plus(j)
        frame1 = frame1.to(device)
        frame1 = frame1.unsqueeze(dim = 0)
        frame2 = frame2.to(device)
        frame2 = frame2.unsqueeze(dim = 0)
        
        anchors = gt_to_anchorpts(pose1, bbox1, device)
        
        if i >= dataset.__len__() // 2: # flip
            anchors = flip_anchors(anchors, flip_over_middle=True)
        
        supervision_frame1.append(frame1)
        supervision_frame2.append(frame2)
        core_gt.append(anchors[0])
        double_gt.append(anchors[1])
        single_gt.append(anchors[2])
    
    supervision_frame1 = torch.cat(supervision_frame1)
    supervision_frame2 = torch.cat(supervision_frame2)
    core_gt = torch.cat(core_gt)
    double_gt = torch.cat(double_gt)
    single_gt = torch.cat(single_gt)

    return supervision_frame1, supervision_frame2, core_gt, double_gt, single_gt

for epoch in range(0, num_epochs):
    print('EPOCH {}:'.format(epoch + 1))
    for batch in dataloader:
        frame1, frame2, idx = batch
        frame1 = frame1.to(device)
        frame2 = frame2.to(device)
        idx = idx.to(device)

        if i % 500 == 0 and i != 0:
            return_imgs = True
        
        if epoch < MILESTONE:
            values = model.train_step(frame1, frame2, return_imgs)
        elif epoch == MILESTONE and core_gt is None:
            values = model.train_step(frame1, frame2, return_imgs)
            supervision_frame1, supervision_frame2, core_gt, double_gt, single_gt = get_gts()
        else: 
            frame1 = torch.cat([frame1, supervision_frame1])
            frame2 = torch.cat([frame2, supervision_frame2])
            values = model.train_step(frame1, frame2, return_imgs, supervision=[core_gt, double_gt, single_gt])
        
        # tensorboard logs, show images every 500 iterations
        if not return_imgs:
            writer.add_scalars(main_tag='losses', tag_scalar_dict=values, global_step=i)
        else:
            values_ = {k: v for k, v in values.items() if len(v.shape) == 0}
            writer.add_scalars(main_tag='losses', tag_scalar_dict=values_, global_step=i)
            for k, v in values.items():
                if len(v.shape) > 1:
                    if k == 'transformed_template' or k == 'transformed_template1':
                        grid = show_images(v, renorm=None)
                    else:
                        grid = show_images(v, renorm=inverse)
                    writer.add_image(k, grid, global_step=i)
        # print(i)
        i += 1
        return_imgs = False
    # save checkpoint
    torch.save({'regressor': model.regressor.state_dict(), 'translator': model.translator.state_dict(),
                'optim': model.optim.state_dict()}, os.path.join(check_dir, 'checkpoint_' + str(i) + '.tar'))