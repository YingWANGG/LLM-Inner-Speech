import torch
from torch.utils.data import Dataset
import pandas as pd
import av
from utils import read_video_pyav
from utils import extractVideo
import numpy as np
class VideoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        # container = av.open(f"/vast/work/public/ml-datasets/ego4d/v1/full_scale/{row[0]}.mp4")
        # frames = read_video_pyav(container, [i for i in range(row.start_frame ,row.end_frame + 1, 15)])
        success, images = extractVideo(f"/vast/work/public/ml-datasets/ego4d/v1/full_scale/{row[0]}.mp4", row[1])
        frames = np.array(images)
        
        if frames.shape[0] < 6:
            # duplicate the last image if unable to load all 6
            diff = int(6 - frames.shape[0])
            frames = torch.tensor(frames)
            frames = torch.cat([frames, frames[-1].unsqueeze(0).repeat(diff, 1, 1, 1)], dim=0)
            
        if self.transform is not None:
            x = self.transform(images=list(frames), text=[row[-1]], return_tensors="pt", padding='max_length')
            pixel_values = x.pixel_values.view(-1, 3, 224, 224)
            label = torch.squeeze(x.input_ids)
            attention_mask = torch.squeeze(x.attention_mask)
            return pixel_values, label, attention_mask, row[-1]
        
        return frames, row.caption


# class Image_Inference_Dataset(Dataset):
#     def __init__(self, df, transform=None):
#         self.df = df
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, index):
#         row = self.df.iloc[index]
        
#         # container = av.open(f"/vast/work/public/ml-datasets/ego4d/v1/full_scale/{row[0]}.mp4")
#         # frames = read_video_pyav(container, [i for i in range(row.start_frame ,row.end_frame + 1, 15)])
#         success, images = extractVideo(f"/vast/work/public/ml-datasets/ego4d/v1/full_scale/{row[0]}.mp4", row[2])
          
#         if self.transform is not None:
#             x = self.transform(images=list(frames), text=[row[-1]], return_tensors="pt", padding='max_length')
#             pixel_values = x.pixel_values.view(-1, 3, 224, 224)
#             label = torch.squeeze(x.input_ids)
#             attention_mask = torch.squeeze(x.attention_mask)
#             return pixel_values, label, attention_mask, row[-1]
        
#         return frames, row.caption