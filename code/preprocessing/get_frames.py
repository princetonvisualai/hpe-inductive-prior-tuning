"""
Step 1: Extract frames from video
"""

import os
import cv2

compute_data_folder = "../../data/"

# get list of all subjects
def get_subjects():
    video_folders = os.listdir(compute_data_folder)[5:]
    return video_folders

def get_video_paths(subject):
    video_paths = os.listdir(compute_data_folder + subject + "/videos/")[2:]
    return video_paths

subjects = get_subjects()

# for each subject
for subject in subjects:
    video_paths = get_video_paths(subject)
    path_prefix = compute_data_folder + subject
    
    save_prefix = compute_data_folder + "h36m/training/" + subject + "/frames/"
    
    f = open("getframeslog.txt", "a")
    f.write("Processing subject " + subject + "\n")
    f.close()

    #print("Processing subject ", subject)
    
    # for each video (i.e. activity), get list of frames

    #if subject == "S1":
    #    video_paths = video_paths[70:]
    
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