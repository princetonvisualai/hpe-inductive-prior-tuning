"""
Analysis Step 1: Collect pose and person bbox diagonal information
"""

import sys
sys.path.append("..")
sys.path.append("../..")

################################################ Imports
OUTPUT_FILE = "analysis.txt"

def write_output(msg, mode = "a"):
    with open("progress/" + OUTPUT_FILE, "a") as f:
        f.write(msg + "\n")
    
write_output("", "w") # clear output file

import torch
import torchvision.transforms as T
import json_tricks as json

from dataset_test import TestImageDataset
import configs.variables as variables

from helper.predict import get_pose

from helper.deepdive import dist

device = torch.device('cuda:0')

################################################ Constants
NUM_JOINTS = 15

PREPROCESSED_TO_ORIG = variables.PREPROCESSED_TO_ORIG # contains the file path of original frame as well

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

##############################################################################################################

# 1. Quantitative Analysis

pose_collection = torch.zeros(DATASET_LENGTH, NUM_JOINTS, 2, device=device)
diagonal_collection = torch.zeros(DATASET_LENGTH, device=device)

for i in range(DATASET_LENGTH):

    if i % 10000 == 0:
        write_output("Working on {}th test frame.".format(i+1))
       
    frame1, frame2, bbox1, pose1, frame1_orig_path = dataset.getitem(i)
            
    # Scale down ground truth pose
    bbox1, pose1 = get_pose(bbox1, pose1)

    bbox1 = torch.tensor(bbox1, device=device)
    pose1 = torch.tensor(pose1, device=device)
    
    ######## COLLECT POSE
    pose_collection[i] = pose1
    
    # PDJ@0.05 = Distance between predicted and true joint < 0.05 * bbox diagonal

    diagonal = dist(bbox1[0], bbox1[1])
    
    ######## COLLECT DIAGONAL
    diagonal_collection[i] = diagonal

torch.save(pose_collection, "pose_collection.pt") # NEED TO DO JUST ONCE
torch.save(diagonal_collection, "diagonal_collection.pt") # NEED TO DO JUST ONCE
