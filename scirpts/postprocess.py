import re
import pandas as pd
import argparse
from utils import load_annotations

def eval_string(raw_s):
    if raw_s == '-1' or '[-1,-1]' in raw_s or raw_s == '[[]]' or raw_s =='[]': return 
    # if the prediction contains less than 2 numbers, it is an invalid prediction
    if len(re.findall(r'\d+', raw_s)) < 2: return 
    # replace the following char by empty string
    for x in [' ', '"', '"']:
        s = raw_s.replace(x, '')
    # replace the following char by a comma
    for x in ['-', '–']:
        s = s.replace(x, '')

    try:
        output = eval(s)
    except: 
        print("FORMAT ERROR:", raw_s, s, flush = True)
        traceback.print_exc()   
    return output


def main(ann_path, raw_path, output_path):
    # load annotation file to dataframe 
    query_df, _ = load_annotations(ann_path)
    query_df = query_df.set_index('query_index')
    # load gpt responses
    response_df = pd.read_csv(raw_path)
    response_df.query_index = pd.to_numeric(response_df.query_index, errors='coerce')
    response_df = response_df.set_index('query_index')
    # join these two dataframes
    response_df = response_df.join(query_df, how='left', lsuffix='l') [['query', 'predictions', 'explanation','cid', 'vid']]
    # process the raw prediction
    response_df['formatted_preds'] = response_df['predictions'].apply(eval_string)
    no_pred_mask = response_df['formatted_preds'].isna()
    response_df['formatted_preds'][no_pred_mask] = '[]' 
    # save csv
    response_df.to_csv(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Postprocess argument parser')
    parser.add_argument('--annotation_path', type=str, help='path to the ego4d nlq annotation file')
    parser.add_argument('--raw_path', type=str, help='path to raw csv file of gpt predictions')
    parser.add_argument('--output_path', type=str, help='path to the output')
    args = parser.parse_args()
    main(args.annotation_path, args.raw_path, args.output_path)

    