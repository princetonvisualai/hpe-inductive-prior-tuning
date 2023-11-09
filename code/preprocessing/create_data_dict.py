"""
Step 2: Create data_dict.json
"""
import sys
sys.path.append("..")
sys.path.append("../..")

import configs.variables as variables

DATA_DIR = variables.DATA_DIR

import os
import numpy as np
import h5py
import json

def get_video_paths(subject):
    video_paths = os.listdir(DATA_DIR + subject + "/videos/")[2:] # TO DO: adjust indexing ([2:]) to filter out any miscellaneous files, like hidden files
    return video_paths

# process bounding boxes of one video
def get_masks_of_video(bbox_path):
    with h5py.File(bbox_path, 'r') as f:
        masks = np.array(f.get("Masks"))
        #refs = np.array(f.get("#refs#"))
        
        return masks[:,0]
        
def get_masked_frame(bbox_path, mask):
    with h5py.File(bbox_path, 'r') as f:
        return np.array(f[mask])

subjects = ["S1", "S11", "S5", "S6", "S7", "S8", "S9"]

data_dict = {}

# for each subject
for subject in subjects:
    #### UPDATE DICT
    data_dict[subject] = {}
    
    video_paths = get_video_paths(subject)
    bbox_paths = [(i[:-4] + ".mat") for i in video_paths]

    bbox_prefix = DATA_DIR + subject + "/bboxes/"
    frame_prefix = DATA_DIR + "h36m/training/" + subject + "/frames/"
    
    f = open("create_data_dict_log.txt", "a")
    f.write("Processing subject " + subject + "\n")
    f.close()
    #print("Processing subject " + subject)
    
    # for each video (activity)
    for j in range(len(bbox_paths)):
        bbox_path = bbox_prefix + bbox_paths[j]
        masks = get_masks_of_video(bbox_path)

        activity = video_paths[j][:-4]
        
        #### UPDATE DICT
        data_dict[subject][activity] = []
        
        save_prefix_vid = frame_prefix + activity + "/"
        
        # for each mask
        for k in range(len(masks)):
            mask = masks[k]
            masked_frame = get_masked_frame(bbox_path, mask).T
            
            bbox = np.where(masked_frame == 1)
            coords = np.array([[bbox[1][0], bbox[0][0]],
                      [bbox[1][-1], bbox[0][-1]]])
            
            coords = {"__ndarray__": [[int(bbox[1][0]), int(bbox[0][0])],
                                      [int(bbox[1][-1]), int(bbox[0][-1])]],
                     "dtype": "int64",
                     "shape": [2, 2],
                     "Corder": True}
            
            #### UPDATE DICT
            data_dict[subject][activity].append({"frame": save_prefix_vid + ("frame%04d.png" % k),
                                                "bounding_box": coords})

with open("data_dict.json", "w") as f:
    json.dump(data_dict, f)