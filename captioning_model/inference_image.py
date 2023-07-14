import torch
import torch.nn as nn
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM
import torch
import json
from utils import *
from torchmetrics.text.rouge import ROUGEScore
from tqdm import tqdm
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BSIZE = 32
num_videos = 50
checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)

model = AutoModelForCausalLM.from_pretrained(checkpoint)
# checkpoint = torch.load("saved_models/GIT_Image_model_step_163394")
# model.load_state_dict(checkpoint)
model.to(device)
# model.train()
model.eval()

def transforms(images,captions):
    inputs = processor(images=images, text=captions, padding="max_length", return_tensors="pt")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs

videos = set(pd.read_csv("videos.csv").iloc[:,0])

val_path = "/scratch/yy2694/ego4d_data/data/v1/annotations/nlq_val.json" 
# val_path = "/scratch/work/public/ml-datasets/ego4d/v2/v2/annotations/nlq_test_unannotated.json"

with open(val_path, 'r') as f:
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

for i in tqdm(range(len(caption_list))):
    vid = caption_list[i][1]
    frame = caption_list[i][-2]
    success, image = extractImages(f"/scratch/work/public/ml-datasets/ego4d/v2/v2/full_scale/{vid}.mp4", frame)
    while not success:
        increment = 1
        success, image = extractImages(f"/scratch/work/public/ml-datasets/ego4d/v2/v2/full_scale/{vid}.mp4", frame + increment)
        if not success:

            success, image = extractImages(f"/scratch/work/public/ml-datasets/ego4d/v2/v2/full_scale/{vid}.mp4", frame - increment)

    if success:
        inputs = transforms(image, ".").to(device)
        pixel_values = inputs.pixel_values
        # labels = inputs.input_ids
        generated_ids = model.generate(pixel_values=pixel_values, max_length=500)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
        caption_list[i][-1] = generated_caption
    
    else:
        print('Failed')


df = pd.DataFrame(caption_list)
df = df.rename(columns={0: 'cid', 1: 'vid', 2: 'timestamp', 3: 'caption'})
try:
    df = df.drop("Unnamed: 0", axis=1)
except:
    pass
df.to_csv(f'nlq_val_{num_videos}_git_image_no_finetune.csv', index=False)
