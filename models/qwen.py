import torch
import transformers
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import os
import copy
import pandas as pd
from tqdm import tqdm
import time
import sys

import numpy as np
import traceback

import warnings
warnings.filterwarnings("ignore")

os.environ['TORCH_COMPILE_UNSUPPORTED']='1'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# PROMPT = """
# The following are multiple choice questions (with exactly one answer).

# {}
# A. curiosity - desire to discover, learn, explore, understand
# B. family - desire for family connection, relatives, family bonds
# C. tranquility - desire for peace, calm, rest, relaxation
# D. vengeance - desire for revenge, retribution, payback
# E. social-contact
# F. romance - desire for romantic love, partnership, intimate relationships
# G. none - doesn't fit any desire option above
# Answer:
# """

GOLD = {
    0:'curiosity',
    1:'family',
    2:'tranquility',
    3:'vengeance',
    4:'social-contact',
    5:'romance',
    6:'none'
}

def safe_cuda_init():
    torch.cuda.synchronize()

    # Enable deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = 42  # or any integer you prefer
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class QwenModel:
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct"):

        safe_cuda_init()

        self.model_name = "Qwen3-VL-8B-Instruct"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, 
            dtype = torch.bfloat16, 
            device_map="auto",
            cache_dir = "/projects/aiwell/conda/envs/ToM/.cache/") #, attn_implementation="flash_attention_2")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = 'left'
        
        # self.model = torch.compile(self.model, mode="reduce-overhead")

    @torch.inference_mode()
    def infer(self, dataloader):
        """
        Run inference on a given dataset
        """
        # Storage for results
        all_results = []
        print("Processing with...")

        st = time.time()
        # Run inference
        for batch in tqdm(dataloader):
            image_paths, prompts, templates, true_labels = batch
            try:
                image_inputs, _ = process_vision_info(templates)
                # Apply chat template
                text = self.processor.apply_chat_template(
                    templates,
                    tokenizer=False, 
                    add_generation_prompt=True
                )
                
                # Prepare all inputs at once
                inputs = self.processor(
                    text=text,
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")
                
                # Generate full text for entire batch
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=300, 
                    do_sample=False,
                    output_logits=True
                )
                # Decode all outputs
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                outputs = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )

                # do it again, but now, we need logits :)
                with torch.no_grad():
                    logit_outputs = self.model(**inputs)

                # logits: (batch_size, seq_len, vocab)
                raw_logits = logit_outputs.logits.cpu().float()

                # take next-token logits for each batch item: (bs, vocab)
                first_token_logits = raw_logits[:, -1, :]

                # target token list
                target_tokens = ["A", "B", "C", "D", "E", "F", "G"]

                # convert to token IDs
                target_token_ids = self.processor.tokenizer.convert_tokens_to_ids(target_tokens)

                option_probs = []  # list of (7,) tensors

                # process each batch element
                for raw_logit in first_token_logits:  # each is (vocab,)
                    probs = torch.nn.functional.softmax(raw_logit, dim=-1)  # (vocab,)
                    position_probs = probs[target_token_ids]  # (7,)
                    option_probs.append(position_probs)


                # Store results for this batch
                for i, (image_path, prompt, true_desire, output, logit) in enumerate(
                    zip(image_paths, prompts, true_labels, outputs, option_probs)
                ):
                    result_dict = {
                        'image_path': image_path,
                        'caption': prompt,
                        'true_desire': true_desire,
                        'result': output.strip(),
                        'option A': float(logit[0]),
                        'option B': float(logit[1]),
                        'option C': float(logit[2]),
                        'option D': float(logit[3]),
                        'option E': float(logit[4]),
                        'option F': float(logit[5]),
                        'option G': float(logit[6])
                    }
                    all_results.append(result_dict)
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        et = time.time()
        print(f"Processing time: {et - st:.2f} seconds")
        
        # Build DataFrame
        df_data = []
                
        for result in all_results:
            # Extract image name from path
            p = result['image_path']
            if p is not None:
                image_name = os.path.basename(result['image_path'])
            else:
                image_name = None
            
            predictions = [result['option A'],
                result['option B'],
                result['option C'],
                result['option D'],
                result['option E'],
                result['option F'],
                result['option G']]
            model_prediction = GOLD[int(np.argmax(predictions))]

            row = [
                image_name,
                result['caption'],
                self.model_name,
                result['true_desire'],
                model_prediction,
                result['result'],
                result['option A'],
                result['option B'],
                result['option C'],
                result['option D'],
                result['option E'],
                result['option F'],
                result['option G']
            ]
            df_data.append(row)
        
        output_df = pd.DataFrame(
            df_data, 
            columns=[
                'ImageName', 
                'Prompt',
                'Model', 
                'True Label', 
                'Predicted Label', 
                'Output Text', 
                'curiosity', 
                'family', 
                'tranquility', 
                'vengeance', 
                'social-contact', 
                'romance', 
                'none'])
        return output_df
