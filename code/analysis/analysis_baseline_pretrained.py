import sys
sys.path.append("..")
sys.path.append("../..")


################################################ Imports
OUTPUT_FILE = "analysis_pretrained.txt"

def write_output(msg, mode = "a"):
    with open("progress/" + OUTPUT_FILE, "a") as f:
        f.write(msg + "\n")
    
write_output("", "w") # clear output file

import torch
import torchvision.transforms as T
import json_tricks as json

from dataset_test import TestImageDataset
import configs.variables as variables

from helper.predict import (get_pose, predict_joints, reorder_anchors)

from helper.deepdive import (dist)

device = torch.device('cuda:0')

################################################ Constants
NUM_JOINTS = 15

VOL_DIR = variables.VOL_DIR
CODE_DIR = variables.CODE_DIR
TEMPLATES_DIR = variables.TEMPLATES_DIR

PREPROCESSED_TO_ORIG = variables.PREPROCESSED_TO_ORIG # contains the file path of original frame as well
CHECKPOINTS_PREFIX = VOL_DIR + "checkpoints_"

TEST_SUBJECTS = ['S9', 'S11']
NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

write_output("Beginning to fetch preprocessed_to_orig mapping.")
# Set up test dataset
# Get mapping from preprocessed frame path to corresponding bbox and pose
with open(PREPROCESSED_TO_ORIG, "r") as f:
    preprocessed_to_orig = json.load(f)

write_output("Done with fetching preprocessed_to_orig mapping.")

dataset = TestImageDataset(transform=NORMALIZE, mapping=preprocessed_to_orig)
DATASET_LENGTH = dataset.length()

write_output("Done with initializing dataset.")

################################################ Load models and configs
from models.pretrained_model import Predictor

predictor = Predictor(batch_size=1, num_parts=18, device=device, 
                 template_path=TEMPLATES_DIR + 'template.json',
                 anchors_path=TEMPLATES_DIR + 'anchor_points.json')

predictor.load_checkpoint(VOL_DIR + 'checkpoints_baseline_pretrained/checkpoint.tar')

write_output("Done with initializing models.")

##############################################################################################################

# 1. Quantitative Analysis
## 1.1 Overall Accuracy

get_midpoint = True

dist1_collection = torch.zeros(DATASET_LENGTH, NUM_JOINTS, device=device)

def predict_joints(predictor, frame1, get_midpoint = False):
    _, anchors = predictor.predict(frame1)
    anchors_reordered = reorder_anchors(anchors, get_midpoint = get_midpoint)
    
    return anchors_reordered

for i in range(DATASET_LENGTH):

    if i % 10000 == 0 or i == 1:
        write_output("Working on {}th test frame.".format(i+1))
       
    frame1, frame2, bbox1, pose1, frame1_orig_path = dataset.getitem(i)
    
    frame1 = frame1.to(device)
    frame1 = frame1.unsqueeze(dim = 0)
    frame2 = frame2.to(device)
    frame2 = frame2.unsqueeze(dim = 0)
            
    # Scale down ground truth pose
    bbox1, pose1 = get_pose(bbox1, pose1)

    bbox1 = torch.tensor(bbox1, device=device)
    pose1 = torch.tensor(pose1, device=device)
    
    ######## COLLECT POSE
    
    # Get predicted pose
    anchors1 = predict_joints(predictor, frame1, get_midpoint = get_midpoint)
    anchors1 = torch.tensor(anchors1, device=device)
        
    ######## COLLECT ERROR FROM GROUND TRUTH
    dist1_collection[i] = dist(pose1, anchors1)

DISTS_DIR = variables.DISTS_DIR

POSTFIX1 = "pretrained"
torch.save(dist1_collection, DISTS_DIR + POSTFIX1 + ".pt")
