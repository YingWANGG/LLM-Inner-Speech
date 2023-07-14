import torch
import torch.nn as nn
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM
import torch
import json
import random 
from utils import *
from evaluate import load
from torchmetrics.text.rouge import ROUGEScore
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1
BSIZE = 16

checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)

model = AutoModelForCausalLM.from_pretrained(checkpoint)

checkpoint = torch.load("saved_models/GIT_Image_model_step_163394")
model.load_state_dict(checkpoint)

model.to(device)
model.train()

rouge = ROUGEScore()


def transforms(images,captions):
    inputs = processor(images=images, text=captions, padding="max_length", return_tensors="pt")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs


narr_list = get_narration_dataset()
random.shuffle(narr_list)
n = len(narr_list)

optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if p.requires_grad], lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=EPOCHS*n)

wandb.init(project="captioning")
wandb.run.name = "GIT_Image_base2"
wandb.watch(model, log="all")



j, update = 0, 0
while j < n:
    k = 0
    images = []
    captions = []
    while k < BSIZE:
        vid, m, caption = narr_list[j]
        j += 1
        success, image = extractImages(f"/vast/work/public/ml-datasets/ego4d/v1/full_scale/{vid}.mp4", m)
        if success: 
            images.append(torch.tensor(image))
            captions.append(caption)
            k+=1
        else:
            print(f"Cannot load {vid}, frame {m}")
            
    inputs = transforms(images, captions).to(device)

    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=500)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    r_score = rouge(generated_caption, captions)

    outs = model(**inputs)
    loss = outs.loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    wandb.log({"loss":loss}, step=j)
    wandb.log({"rouge1_fmeasure": r_score["rouge1_fmeasure"]}, step=j)
    wandb.log({"rouge1_precision":r_score["rouge1_precision"]}, step=j)
    wandb.log({"rouge1_recall":r_score["rouge1_recall"]}, step=j)
    wandb.log({"rougeL_fmeasure":r_score["rougeL_fmeasure"]}, step=j)
    wandb.log({"rougeL_precision":r_score["rougeL_precision"]}, step=j)
    wandb.log({"rougeL_recall":r_score["rougeL_recall"]}, step=j)
    update +=1

    # generated_ids = model.generate(pixel_values=pixel_values, max_length=500)
    # generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # generated_captions.append(generated_caption)
    if update%(500) == 0 or update == 1:
        torch.save(model.state_dict(), f"/scratch/ds5749/GIT/GIT_Image_model_epoch2_step_{j}")