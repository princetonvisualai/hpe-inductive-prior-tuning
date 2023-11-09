"""
Step 1: Extract frames from video
"""
import sys
sys.path.append("..")
sys.path.append("../..")

import configs.variables as variables

import os
import cv2

DATA_DIR = variables.DATA_DIR

def get_video_paths(subject):
    video_paths = os.listdir(DATA_DIR + subject + "/videos/")[2:]  # TO DO: adjust indexing ([2:]) to filter out any miscellaneous files, like hidden files
    return video_paths

subjects = ["S1", "S11", "S5", "S6", "S7", "S8", "S9"]

# for each subject
for subject in subjects:
    video_paths = get_video_paths(subject)
    path_prefix = DATA_DIR + subject
    
    save_prefix = DATA_DIR + "h36m/training/" + subject + "/frames/"
    
    f = open("getframeslog.txt", "a")
    f.write("Processing subject " + subject + "\n")
    f.close()

    #print("Processing subject ", subject)
    
    # for each video (i.e. activity), get list of frames
    
    for j in range(len(video_paths)):
        #if j % 3 == 0:
        #    print("----video ", (j+1) / len(video_paths))
        
        video_path = path_prefix + "/videos/" + video_paths[j]
        activity = video_paths[j][:-4]
        
        save_prefix_vid = save_prefix + activity + "/"
        
        # get frames for this video
        vidcap = cv2.VideoCapture(video_path)
        i = 0
        while True:
            success,image = vidcap.read()
            if not success:
                break
            cv2.imwrite(save_prefix_vid + ("frame%04d.png" % i), image)     # save frame as JPEG file
            i += 1