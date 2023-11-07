"""
Step 5: Create mapping from preprocessed frame to original frame
    - Creates preprocessed_to_orig.json
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import matplotlib.pyplot as plt
import json_tricks as json
import cdflib

plt.rcParams["savefig.bbox"] = "tight"

########################
# CONSTANTS

PREFIX = "../../"
PREPROCESSED_TO_ORIG = "preprocessed_to_orig.json" # contains the file path of original frame as well
########################

"""
Create preprocessed to original frame, bbox, pose mapping
"""
def create_mapping():

    # get original bounding box and pose information
    with open(PREFIX + "code/preprocessing/data_dict.json", "r") as f:
        data_dict = json.load(f)

    # get original bounding box and pose information
    with open(PREFIX + "code/preprocessing/mapping_preprocessed.json", "r") as f:
        mapping = json.load(f)

    for i in mapping:
        mapping[i] = PREFIX + "data/preprocessed/" + mapping[i][25:]
    
    preprocessed_to_orig = {}

    for s in ['S9', 'S11']:
        subject = data_dict[s]

        for video in subject.keys():

            frames = subject[video]

            # if this video was preprocessed
            if frames[0]['frame'] in mapping:
                poses = cdflib.CDF(PREFIX + "data/poses/" + s + "/" + video + ".cdf")
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
    with open(PREPROCESSED_TO_ORIG, "w") as f:
         json.dump(preprocessed_to_orig, f)
    