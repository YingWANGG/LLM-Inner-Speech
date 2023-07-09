import pandas as pd
import openai
import os
from tqdm import tqdm
import sys
import openai
import traceback
import io
import argparse
from utils import load_annotations

SEP = '\n\n###\n\n'
S1 = "You are the person #C and other person is denoted as #O. Here is a list of pairs, each includes an interval of timestamps \
and a summary of activities that you did in your memory at those timestamps. Imagine you want to recall your memory \
to answer a list of questions below:"
S2 = "However, just looking at the summaries perhaps will not help you answer the question. Now, imagine that you can go back in time to revisit a temporal window \
consisting of timestamps and see the visual scene again. Make your best guess, what are the time intervals that you would like to revisit?  Think step by step. \
Output the above results into a TSV, where the column names are query_index, question, predictions, explanation. \
Namely, the first column is the query index, the second column is the question, the third column is the list of predicted intervals (e.g. [[a,b],[c,d]...), and the fourth column is the explanation. \
Could you try to make your best guesses using your reasoning skills? If you are unconfident about your prediction, you can guess multiple wider intervals. \
The resulting TSV is "
SYSTEM = "You are the person #C and want to recall your memory to answer a list of questions."

def get_prompt(captions, queries):
    return SEP + captions + SEP + S1 + SEP + queries + SEP + S2 + SEP

def preprocess(cap_df, clip_timestamp_dict, sample_freq = 100):
    union_rows = []
    grouped_cap_df = cap_df.groupby(['vid','cid'])
    for (vid, cid), clip_df in grouped_cap_df:
        row = clip_df.iloc[0]
        caption, timelist = row['caption'], [row['timestamp']]   
        clip_start, clip_end = clip_timestamp_dict[(vid,cid)] # start & end timestamp
        for i, row in clip_df[['timestamp','caption']].iterrows():
            # if the caption is the same as the previous one, or the current length is 0
            # these two timestamps should be merged
            if row['caption'] == caption or len(caption) == 0:
                timelist.append(row['timestamp'])
            else:
                # record the previous caption and related timestamps
                # we expand the timestamps by half sample_freq of the captioning model
                # so the union of all resulting intervals will be the same as [clip_start, clip_end]
                start = max(int(timelist[0]) - sample_freq//2, clip_start)
                end = min(int(timelist[-1]) + sample_freq//2, clip_end)
                union_rows.append({'vid':vid, 'cid':cid, 'caption': caption, 'timestamp': f"{str(start)}-{str(end)}", 'timestamp_list':timelist})
                # current caption and timestamp
                caption, timelist = row['caption'], [row['timestamp']]
        # record the last caption and related timestamps
        # the end should always be the clip_end
        start = max(int(timelist[0]) - sample_freq//2, clip_start)
        end = clip_end
        union_rows.append({'vid':vid, 'cid':cid, 'caption': caption, 'timestamp': f"{str(start)}-{str(end)}", 'timestamp_list':timelist})
    union_df = pd.DataFrame(union_rows)  
    return union_df

def gpt_inference(union_df, query_df):
    response_dict = {}
    union_df = union_df.groupby(['vid','cid'])
    query_df = query_df.groupby(['vid','cid'])
    response_df = pd.DataFrame(columns=['query_index', 'question', 'predictions', 'explanation'])
    for vid, cid in list(union_df.groups):
        clip_df = union_df.get_group((vid,cid))
        captions = clip_df[['timestamp', 'caption']].to_string(index=False)
        clip_query_df = query_df.get_group((vid,cid))
        queries = clip_query_df[['query_index', 'query']].to_string(index=False)
        s = get_prompt(captions, queries)
        response = ""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": s}
                    ]
            )
        except:
            print(f"OPENAI ERROR: {vid}, {cid}", flush=True)
            traceback.print_exc()
            continue
        response_dict[(vid,cid)] = response
        try:
            data = response["choices"][0]["message"]["content"]
            result_df = pd.read_csv(io.StringIO(data), sep='\t')
            response_df = pd.concat([response_df, result_df], ignore_index=True)
        except:
            print(f"FORMAT ERROR: {vid}, {cid}", flush=True)
            print(response)
            traceback.print_exc()
        break
    return response_df

def main(ann_path, caption_path, output_path):
    query_df, clip_timestamp_dict = load_annotations(ann_path)
    cap_df = pd.read_csv(caption_path)
    preprocessed_cap_df = preprocess(cap_df, clip_timestamp_dict, sample_freq = 100)
    response_df = gpt_inference(preprocessed_cap_df, query_df)
    response_df.to_csv(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LLM inner speech argument parser')
    parser.add_argument('--annotation_path', type=str, help='path to the ego4d nlq annotation file')
    parser.add_argument('--caption_path', type=str, help='path to the captions from the egocentric video')
    parser.add_argument('--output_path', type=str, help='path to the csv containing the results from GPT4')
    parser.add_argument('--openai_key', type=str, help='openai api key')
    args = parser.parse_args()
    openai.api_key =  args.openai_key
    main(args.annotation_path, args.caption_path, args.output_path)

