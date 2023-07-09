# LLM-Inner-Speech
The third-place solution (VioletPanda) to the Ego4d NLQ Challenge 2023.

## Overview
We present a novel solution to the Ego4D NLQ challenge, inspired by the concept of "inner speech" in cognitive science. Our proposed pipeline first uses image and video captioning models to generate captions that encapsulate sufficient details from the egocentric video. The captions are then fed into a large language model (LLM) to generate coarse-grained predictions containing multiple potential response windows, and these predictions are further refined by a pre-trained NLQ model. For more details, please read our [technical report](https://github.com/YingWANGG/LLM-Inner-Speech/blob/main/LLM_inner_speech.pdf).

## Quick Start

```
# Preprocess the data and use OpenAI's GPT4 to predict temporal windows given captions and queries.
python gpt4_inf.py
--annotation_path <the path to the official Ego4D annotation file>
--caption_path <the path to the csv containing captions from the egocentric videos, which contains 4 columns: cid, vid, timestamp, caption>
--output_path <the path to the csv containing the responses from GPT>
--openai_key <your OpenAi API key>

# Postprocess the response file of GPT
python postprocess.py
--annotation_path <the path to the official Ego4D annotation file>
--raw_path <the path to the csv containing the raw responses from GPT>
--output_path <the path to the csv containing the preprocessed predictions>
```
