"""
script for processing raw H36m images. The folder structure for the extracted frames should be:
h36m
    +-- training
        +-- subject
            +-- frames
                +-- activity
                    +-- frame*.png

inputs:
    subj [string]
    path to json file containing dictionary with metadata with the following structure:
    {subj:
        +-- activity [list]
            +-- {frame: [str] path to frame,
                 bounding_box: numpy array [[column_topleft [int], row_topleft], [column_bottomright, row_bottomright]]
                    }
            }

if unclear, have a look at the data_dict.json file in the processing folder
"""

# python h36m_processing.py --subj S1 --meta data_dict.json --dst ../data/preprocessed/training

dst = "../data/preprocessed/training"
subjects = ["S1", "S11", "S5", "S6", "S7", "S8", "S9"]
meta = "data_dict.json"

import cv2
import json_tricks as json
import os
import numpy as np
from transforms import get_affine_transform
import random
import string

####################
with open("preprocessinglog.txt", "a") as g:
    g.write("Starting to load data_dict.json\n")

# load metadata
with open(meta, 'r') as file:
    center_dict = json.load(file)

####################
with open("preprocessinglog.txt", "a") as g:
    g.write("Finished loading data_dict.json\n")

# vertical scale parameter for affine transformation, check get_affine_transform.py for details
# you might have to play around with this a bit. (effectively it's a zoom factor)
scale_factor = 170

# function for transforming frame
def transform_frame(data, center, scale):
    trans = get_affine_transform(center=center, scale=scale, rot=0, output_size=np.array((256, 256)))
    transformed = cv2.warpAffine(
        data,
        trans,
        # hard-coded, change if necessary
        (256, 256),
        flags=cv2.INTER_LINEAR)

    return transformed

# retrieve bounding box coordinates from string
def get_bbx(mystr):
    bbx = np.asarray([int(float(s)) * scale_factor for s in mystr.split(',')]).reshape((2, 2))

    return bbx


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


valid_activities = ['Waiting', 'Posing', 'Greeting', 'Directions', 'Discussion', 'Walking', 'Greeting', 'Photo',
                    'Purchases', 'WalkTogether', 'WalkDog']

mapping_preprocessed = {}

for subj in subjects:

    ####################
    with open("preprocessinglog.txt", "a") as g:
        g.write("Preprocessing " + subj + "\n")

    activities = list(center_dict[subj].keys())

    # raster_positions = []

    # cameras = []

    destination_path = os.path.join(dst, subj)

    for act in activities:

        name = ''.join([s for s in act if not s.isdigit()]).replace('.', '')
        if name not in valid_activities:
            continue
        entries = center_dict[subj][act]
        bboxes = [entry['bounding_box'] for entry in entries]
        frames = [entry['frame'] for entry in entries]
        # define bounding boxes
        # bbx_dict = {}
        for bbox, frame in zip(bboxes, frames):
            
            # divide image into a grid where each bounding box falls into a certain region
            top_left = np.floor(bbox[0] / scale_factor)
            bottom_right = np.ceil(bbox[1] / scale_factor)
            # define folder for every region, ie. all frames in that folder will share the same bounding box and therefore
            # background (across activities)
            raster_pos = ','.join([str(top_left[0]), str(top_left[1]), str(bottom_right[0]), str(bottom_right[1])])
            # load and transform frame
            data = cv2.imread(frame)
            bbx = get_bbx(raster_pos)
            scale = ((bbx[1][1] - bbx[0][1]) / scale_factor)
            center = np.asarray((bbx[0, 0] + (bbx[1, 0] - bbx[0, 0]) // 2, bbx[0, 1] + (bbx[1, 1] - bbx[0, 1]) // 2))
            scale = np.asarray([scale, scale])
            transformed = transform_frame(data, center, scale)
            # frame_id = os.path.split(frame)[1]
            # find camera id
            cam_id = frame.split('/')[-2].split('.')[1]
            final_path = os.path.join(destination_path, raster_pos + '_' + cam_id)
            saved_path = final_path, 'frame_' + id_generator(size=10) + '.png'
            cv2.imwrite(os.path.join(saved_path), transformed)
            mapping_preprocessed[frame] = saved_path

with open("mapping_preprocessed.json", "w") as f:
    json.dump(mapping_preprocessed, f)


