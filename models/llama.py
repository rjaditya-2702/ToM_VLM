import os
import torch
import transformers
transformers.set_seed(42)
from transformers import MllamaForConditionalGeneration, AutoProcessor
# from .dataloader import DataSetLoader
from torch.utils.data import DataLoader
from PIL import Image
import time
from tqdm import tqdm
import pandas as pd
import sys
from torch.utils.data import Dataset
import copy

import numpy as np
import traceback

import warnings
warnings.filterwarnings("ignore")

os.environ['TORCH_COMPILE_UNSUPPORTED']='1'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

class LlamaModel:
    def __init__(self, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
        
        safe_cuda_init()        
        self.model_name = "Llama-3.2-11B-Vision-Instruct"  # You can make this configurable
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            dtype = torch.bfloat16,
            device_map = 'auto'
        )
        
        # self.model = torch.compile(self.model, mode="reduce-overhead")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = 'left'
        
    @torch.inference_mode()
    def infer(self, dataloder):
        # Storage for results
        all_results = []
        
        print("Processing with images...")
        st = time.time()
        
        # Run inference for image inputs
        for batch in tqdm(dataloder):
            # Unpack the batch tuple
            image_paths, prompts, templates, true_labels = batch
            images = [[Image.open(p)] if p is not None else None for p in image_paths]
            
            try:                
                # Prepare all inputs at once

                text = self.processor.apply_chat_template(
                    templates,
                    tokenizer=False, 
                    add_generation_prompt=True
                )

                inputs = self.processor(
                    text=text,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")
                
                # Generate for entire batch
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=300, 
                    do_sample=False
                )
                
                # Decode all outputs
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                outputs = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False,
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
        print(f"Image processing time: {et - st:.2f} seconds")
        
        # Build DataFrame
        df_data = []
        
        for result in all_results:
            # Extract image name from path
            image_name = os.path.basename(result['image_path'])
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