import cv2
import json
import re
import numpy as np
# import av
import os
import torch.distributed as dist
import torch 

def extractImages(pathIn, start):
    vidcap = cv2.VideoCapture(pathIn)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,start)    
    success,image = vidcap.read()   
    return success,image

def extractVideo(pathIn, start):
    images = []
    
    if start < 45:
        frames =  np.array([start+i*15 for i in range(6)])
    else:
        frames =  np.array([start+i*15 for i in range(-3, 3)])

    for i in frames:
        vidcap = cv2.VideoCapture(pathIn)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES,i)    
        success,image = vidcap.read()   
        if success:
            images.append(image)
        else:
            print(f'Failed to extract {pathIn} frame {i}')
    return len(images)>0 , images

def get_narration_dataset(train=True, video=False):
    narration_path = "/vast/work/public/ml-datasets/ego4d/v1/annotations/"
    with open(narration_path+"narration.json") as json_file:
        narration_data = json.load(json_file)
        
    val_path = "/scratch/yy2694/ego4d_data/data/v1/annotations/nlq_val.json"  
    with open(val_path, 'r') as f:
        val_ann = json.load(f)

    val_set = set(v['video_uid'] for v in val_ann['videos'])
    pass_counter = 0
    narr_list = []
    for k, v in narration_data.items():
        if train:
            if k in val_set: 
                pass_counter += 1
                continue
            elif 'narration_pass_1' in v and os.path.isfile(f"/vast/work/public/ml-datasets/ego4d/v1/full_scale/{k}.mp4"):
                for n in v['narration_pass_1']['narrations']:
                    if '#Unsure' in n['narration_text']: continue
                    if '#unsure' in n['narration_text']: continue
                    narration_text = re.sub(r"#\S+", "", n['narration_text'])
                    # if video:
                    #     frames = sample_caption_indices(n['timestamp_frame'])
                    #     start, end = min(frames), max(frames)
                    #     narr_list.append((k, start, end, narration_text))
                    # else:
                    narr_list.append((k, n['timestamp_frame'], narration_text))
            
        else:
            if k in val_set: 
                if 'narration_pass_1' in v:
                    for n in v['narration_pass_1']['narrations']:
                        if '#Unsure' in n['narration_text']: continue
                        if '#unsure' in n['narration_text']: continue
                        narration_text = re.sub(r"#\S+", "", n['narration_text'])
                        narr_list.append((k, n['timestamp_frame'], narration_text))
            else:
                pass_counter+=1
            
    n = len(narr_list)
    print(f"Pass {pass_counter} videos; Number of training captions: {n}")
    return narr_list

def sample_caption_indices(frame):
    if frame < 4:
        return np.array([frame+i*15 for i in range(6)])
    return np.array([frame+i*15 for i in range(-3, 3)])

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def init_distributed(port=40111, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            logger.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ['MASTER_PORT'] = str(port)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank)
    except Exception:
        world_size, rank = 1, 0
        logger.info('distributed training not available')

    return world_size, rank


# def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
#      converted_len = int(clip_len * frame_sample_rate)
#      end_idx = np.random.randint(converted_len, seg_len)
#      start_idx = end_idx - converted_len
#      indices = np.linspace(start_idx, end_idx, num=clip_len)
#      indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
#      return indices

# import os
# import glob
# from collections import Counter
# folder_path = '/vast/work/public/ml-datasets/ego4d/v1/full_scale/'

# # join folder path with the search pattern
# search_pattern = os.path.join(folder_path, "*.mp4")

# # use glob to find all files with the search pattern
# mp4_files = glob.glob(search_pattern)

# # print the list of mp4 files
# # print(mp4_files)


# fps_all = []
# for file in mp4_files:
#     cam = cv2.VideoCapture(file)
#     fps = cam.get(cv2.CAP_PROP_FPS)
#     fps_all.append(fps)
    
# counts = Counter(fps_all)
# for value, count in counts.items():
#     print(f"{value}: {count}") 