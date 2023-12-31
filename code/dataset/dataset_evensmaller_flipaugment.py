import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import configs.variables as variables

DATA_DIR = variables.DATA_DIR

class ImageDataset(Dataset):
    """Implement your map-style dataset class here!"""
    
    def __init__(self, transform=None):
        #subjects = ["S1", "S11", "S5", "S6", "S7", "S8", "S9"]
        subjects = ["S1", "S5", "S6", "S7", "S8"]
        self.transform = transform
        self.frame_paths = []
        self.len = 0
        
        for subject in subjects:
            subject_prefix = DATA_DIR + "preprocessed/training/" + subject + "/"
            folders = os.listdir(subject_prefix)
            for folder in folders:
                if not folder.startswith("."):
                    bbox_prefix = subject_prefix + folder
                    frames = os.listdir(bbox_prefix)
                    
                    # filter out non pngs
                    frames = np.array(frames)
                    frames = frames[np.char.endswith(frames, '.png')].tolist()
                    frames = sorted(frames)
                    frames = [(bbox_prefix + "/" + frame) for frame in frames]
                    
                    if len(frames) > 1:
                        if (len(frames) // 2) % 2 == 0: # if result after dividing by 2 is even
                            length = len(frames) // 2
                        else:
                            length = (len(frames) // 2) + 1
                            
                        self.frame_paths += frames[0:length]
                        self.len += (length // 2)
                
        print("Length of Dataset: ", self.len * 2)

        print("Finished initializing dataset.")

                
    def __len__(self):
        return self.len * 2

    def __getitem__(self, idx):

        idx_new = idx % (self.len)

        frame1 = read_image(self.frame_paths[2 * idx_new])[[2,1,0],:,:]
        frame2 = read_image(self.frame_paths[2 * idx_new + 1])[[2,1,0],:,:]

        if idx >= self.len:
            frame1 = torch.flip(frame1, (2,))
            frame2 = torch.flip(frame2, (2,))

        frame1 = frame1.div(255)
        frame2 = frame2.div(255)
        
        if self.transform is not None:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
            
        return frame1, frame2

        
