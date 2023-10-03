import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../..")

import json_tricks as json
import cdflib

prefix = "/scratch/network/nobliney/project/"

# get original bounding box and pose information
with open(prefix + "processing/data_dict.json", "r") as f:
    data_dict = json.load(f)

print("Done getting data_dict")

# get original bounding box and pose information
with open(prefix + "data/mapping_preprocessed.json", "r") as f:
    mapping = json.load(f)

print("Done getting mapping")

for i in mapping:
    mapping[i] = "/scratch/network/nobliney/project/data/preprocessed/" + mapping[i][25:]

preprocessed_to_orig = {}

for s in ['S1', 'S11', 'S5', 'S6', 'S7', 'S8', 'S9']:
    print(s)
    subject = data_dict[s]
        
    for video in subject.keys():
        
        frames = subject[video]
        
        # if this video was preprocessed
        if frames[0]['frame'] in mapping:
            poses = cdflib.CDF(prefix + "data/poses/" + s + "/" + video + ".cdf")
            poses = poses['Pose'].squeeze()

            for i in range(len(frames)):  
                path = frames[i]['frame']
                # preprocessed frame --> bounding box, pose
                preprocessed_to_orig[mapping[path]] = {'bbox': frames[i]['bounding_box'], 
                                                       'pose': poses[i],
                                                      'path': path}
        else:
            continue

# Get mapping from preprocessed frame path to corresponding bbox and pose
PREPROCESSED_TO_ORIG = "preprocessed_to_orig_all.json"

print("Saving map")
with open(PREPROCESSED_TO_ORIG, "w") as f:
    json.dump(preprocessed_to_orig, f)