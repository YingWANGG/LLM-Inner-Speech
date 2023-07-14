import torch
import torch.nn as nn
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM
import json
import random 
from utils import *
from torchmetrics.text.rouge import ROUGEScore
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = "microsoft/git-base-vatex"
processor = AutoProcessor.from_pretrained(checkpoint)

model = AutoModelForCausalLM.from_pretrained(checkpoint)
# checkpoint = torch.load("GIT_Video_model_62000")
# model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print()

videos = set(pd.read_csv("videos.csv").iloc[:,0])
# val_ann_path = "/scratch/work/public/ml-datasets/ego4d/v2/v2/annotations/nlq_test_unannotated.json"
val_ann_path = "/scratch/yy2694/ego4d_data/data/v1/annotations/nlq_val.json" 

with open(val_ann_path, 'r') as f:
    val_ann = json.load(f)
    
caption_list = []
for i in range(len(val_ann['videos'])):
    if val_ann['videos'][i]['video_uid'] in videos:
        for j in range(len(val_ann['videos'][i]['clips'])):
            clip = val_ann['videos'][i]['clips'][j]
            start_frame = clip['video_start_frame']
            while start_frame <= clip['video_end_frame']:
                caption_list.append([clip['clip_uid'], val_ann['videos'][i]['video_uid'], start_frame, ''])
                start_frame+=100
df = pd.DataFrame(caption_list)

from torch.utils.data import Dataset, DataLoader

class Image_Inference_Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        success, images = extractVideo(f"/scratch/work/public/ml-datasets/ego4d/v2/v2/full_scale/{row[1]}.mp4", row[2])
        frames = np.array(images)
        if frames.shape[0] < 6:
            # duplicate the last image if unable to load all 6
            if frames.shape[0] == 0:
                print(frames.shape[0])
                frames = torch.zeros((6, 3, 224, 224))
            else:
                diff = int(6 - frames.shape[0])
                frames = torch.tensor(frames)
                frames = torch.cat([frames, frames[-1].unsqueeze(0).repeat(diff, 1, 1, 1)], dim=0)
            
        # if self.transform is not None:
        # x = self.processor(images=list(frames), text=[row[-1]], return_tensors="pt", padding='max_length')
        x = processor(images=list(frames), return_tensors="pt", padding='max_length')
        pixel_values = x.pixel_values.view(-1, 3, 224, 224)
        # label = torch.squeeze(x.input_ids)
        # attention_mask = torch.squeeze(x.attention_mask)
        # return pixel_values, attention_mask
        return pixel_values

dataset = Image_Inference_Dataset(df)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=12)
print('dataloader length', len(dataloader))
captions = []
for i, batch in tqdm(enumerate(dataloader)):
    inputs = batch.to(device)
    generated_ids = model.generate(pixel_values=inputs, max_length=500)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    for words in generated_caption:
        captions.append(words)
        
df[3] = captions
df = df.rename(columns={0: 'cid', 1: 'vid', 2: 'timestamp', 3: 'caption'})
df.to_csv(f'nlq_val_50_git_video_no_finetune.csv', index=False)
