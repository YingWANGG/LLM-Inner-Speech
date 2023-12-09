import json
import cv2
import pandas as pd

def extractImages(pathIn, timestamp):
    vidcap = cv2.VideoCapture(pathIn)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, timestamp)    
    success, image = vidcap.read()   
    return success, image

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_annotations(ann_path, mode = "test"):
    query_rows = []
    clip_timestamp_dict = {}
    count = 0
    annotations = load_json(ann_path)
    for v in annotations['videos']:
        vid = v['video_uid']
        for c in v['clips']:
            cid = c['clip_uid']
            # start and end timestamp of a clip
            # use start_sec * fps and end_sec * fps because some start_timestamp and end_timestamp are the same in the test set
            clip_timestamp_dict[(vid,cid)] = (round(c['video_start_sec']*30), round(c['video_end_sec']*30))
            for a in c['annotations']:
                for query in a['language_queries'] :
                    if 'query' in query and query['query']:
                        count += 1
                        if mode == "test":
                            query_rows.append({'vid':vid, 'cid':cid, 'query': query['query'], 'query_index': count})
                        else:
                            # also include the gt window
                            query_rows.append({'vid':vid, 'cid':cid, 'query': query['query'], 'query_index': count, 'gt_start': query['video_start_frame'], 'gt_end': query['video_end_frame']})
    query_df = pd.DataFrame(query_rows)
    return query_df, clip_timestamp_dict