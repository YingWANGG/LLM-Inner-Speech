import torch
import torch.nn as nn
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM

import torch.multiprocessing as mp

import pandas as pd
import numpy as np

import json
import random 
import os

from utils import *
from dataset import VideoDataset
from torch.utils.data.distributed import DistributedSampler

from torchmetrics.text.rouge import ROUGEScore
from torch.utils.data import Dataset, DataLoader
# import av

import wandb

def train(ddp):
    
    if ddp:
        local_rank = 0
        device = torch.device("cuda:{}".format(local_rank))
        torch.distributed.init_process_group(backend="nccl")
        dist.init_process_group(backend='nccl', init_method='env://')
    
    
    batch_size = 8

    processor = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex")
    checkpoint = torch.load("/scratch/ds5749/GIT/GIT_Video_model_39000")
    model.load_state_dict(checkpoint)

    if ddp:
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    # elif torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     torch.cuda.set_device('cuda:0')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)



    narr_list = get_narration_dataset(train=True, video=True)
    n = len(narr_list)
    df = pd.DataFrame(narr_list)
    df = df.rename(columns={0: 'VideoID', 1: 'start_frame', 2: 'end_frame', 3: 'caption'})

    df = df.iloc[batch_size*39000 + 1: , :]
    dataset = VideoDataset(df, processor)

    # if torch.cuda.device_count() > 1:
    #     train_sampler = DistributedSampler(dataset=dataset, shuffle=True)                                                  
    #     trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=10, pin_memory=True)
    # else:
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    # def my_collate_fn(data):
    #     return tuple(data)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=my_collate_fn)
    # def my_collate_fn(data):
    #     return tuple(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    print('dataloader length', len(dataloader))
    # from tqdm import tqdm
    # for i, batch in tqdm(enumerate(dataloader)):
    #     pass

    optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if p.requires_grad], lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=n)

    rouge = ROUGEScore()

    wandb.init(project="captioning")
    wandb.run.name = "GIT_Video"
    wandb.watch(model, log="all")

#     if torch.cuda.device_count() > 1:
#         trainloader.sampler.set_epoch(1)
        
    for i, batch in enumerate(dataloader):
        
        # pixel_values = batch[0].to(device)
        # input_ids = batch[1].to(device)
        # attention_mask = batch[2].cuda()
        pixel_values = batch[0].cuda()
        input_ids = batch[1].cuda()
        
        optimizer.zero_grad()
        
        outs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=batch[2], labels=input_ids)
        loss = outs.loss
        
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
        r_score = rouge(generated_caption, batch[-1])
        
        wandb.log({"loss":loss}, step=i)
        wandb.log({"rouge1_fmeasure": r_score["rouge1_fmeasure"]}, step=i)
        wandb.log({"rouge1_precision":r_score["rouge1_precision"]}, step=i)
        wandb.log({"rouge1_recall":r_score["rouge1_recall"]}, step=i)
        wandb.log({"rougeL_fmeasure":r_score["rougeL_fmeasure"]}, step=i)
        wandb.log({"rougeL_precision":r_score["rougeL_precision"]}, step=i)
        wandb.log({"rougeL_recall":r_score["rougeL_recall"]}, step=i)

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i%1000 == 0:
            torch.save(model.state_dict(),f"/scratch/ds5749/GIT/GIT_Video_model_{i + 39000}")
            
if __name__ == "__main__":
    train(ddp=False)