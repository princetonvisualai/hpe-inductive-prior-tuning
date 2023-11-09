import sys
sys.path.append("..")
sys.path.append("../..")

################################################ Imports
OUTPUT_FILE = "analysis_var_constrained.txt"

def write_output(msg, mode = "a"):
    with open("progress/" + OUTPUT_FILE, "a") as f:
        f.write(msg + "\n")
    
write_output("", "w") # clear output file

import torch
import torchvision.transforms as T
import json_tricks as json

from utils.helper import load_config
from dataset_test import TestImageDataset
import configs.variables as variables

from helper.predict import (adjust_configs, most_recent_checkpoint, load_checkpoint, predict_joints)

from helper.deepdive import (dist)

device = torch.device('cuda:0')

################################################ Constants
NUM_JOINTS = 15

VOL_DIR = variables.VOL_DIR
CONFIGS_DIR = variables.CONFIGS_DIR
CODE_DIR = variables.CODE_DIR

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
from models.model_constrained import Model as model1

POSTFIX1 = "constrained"

CONFIG1_YAML = POSTFIX1 + ".yaml"

CHECKPOINTS_FOLDER1 = CHECKPOINTS_PREFIX + POSTFIX1 + "/"

################################################

# Get configurations of both models
config1 = load_config(CONFIGS_DIR + CONFIG1_YAML)
# train --> test configurations
adjust_configs(config1)

# Instantiate both predictors
predictor1 = model1(config1, device=device)
predictor1.regressor = predictor1.regressor.eval()
predictor1.translator = predictor1.translator.eval()

# Load checkpoint
checkpoint1 = most_recent_checkpoint(CHECKPOINTS_FOLDER1)

load_checkpoint(predictor1, CHECKPOINTS_FOLDER1 + checkpoint1)

write_output("Done with initializing models.")

##############################################################################################################

# 1. Quantitative Analysis

get_midpoint = True

NUM_PROPORTIONS = 13

var1_collection = torch.zeros(DATASET_LENGTH, NUM_PROPORTIONS, device=device)

for i in range(DATASET_LENGTH):

    if i % 10000 == 0:
        write_output("Working on {}th test frame.".format(i+1))
       
    frame1, frame2, bbox1, pose1, frame1_orig_path = dataset.getitem(i)
    
    frame1 = frame1.to(device)
    frame1 = frame1.unsqueeze(dim = 0)
    frame2 = frame2.to(device)
    frame2 = frame2.unsqueeze(dim = 0)
    
    # Get predicted pose
    anchors1 = predict_joints(predictor1, frame1, frame2, get_midpoint = get_midpoint)

    anchors1 = torch.tensor(anchors1, device=device)

    anchors1_endpoints = torch.ones(NUM_JOINTS, 2).to(device)
    anchors1_endpoints[:,:] = anchors1[[8,0,1,2,0,4,5,7,8,7,9,10,7,12,13]]
    
    ######## COLLECT DIAGONAL, ERROR FROM GROUND TRUTH
    dists1 = dist(anchors1_endpoints, anchors1)[[0,1,2,3,4,5,6,9,10,11,12,13,14]]
    var1_collection[i] = (dists1 / dists1[0])

VAR_DIR = variables.VAR_DIR

torch.save(var1_collection, VAR_DIR + POSTFIX1 + ".pt")
