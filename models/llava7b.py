import transformers
transformers.set_seed(42)
from transformers import pipeline
import re
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import time
import os
import gc

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
       
class Llava7BModel:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf"):
        
        safe_cuda_init()

        self.pipe = pipeline("image-text-to-text", model=model_id, device_map = 'auto')        
        self.pipe.tokenizer.padding_side = 'left'
        if self.pipe.tokenizer.pad_token is None:
            self.pipe.tokenizer.pad_token = self.pipe.tokenizer.eos_token
        
        torch.cuda.synchronize()
        # self.pipe.model = torch.compile(self.pipe.model, mode="reduce-overhead")

        self.model_name = "llava-hf/llava-1.5-7b-hf"
    
    @torch.inference_mode()
    def infer(self, dataloader):
        
        # Storage for results
        all_results = []
        print("Processing with images...")
        st = time.time()
        
        # Run inference for image inputs
        try:
            for batch in tqdm(dataloader):
                # Unpack the batch tuple
                image_paths, prompts, templates, true_labels = batch

                batch_outputs = self.pipe(
                    text=templates,  # List of all messages in the batch
                    max_new_tokens=300,
                    do_sample=False,
                    batch_size=len(templates)  # Specify batch size
                )

                # do it again, but now, we need logits :)
                with torch.no_grad():
                    logit_outputs = self.pipe.model(**inputs)
                    
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
                    zip(image_paths, prompts, true_labels, batch_outputs, option_probs)
                ):
                    # Extract caption from messages
                    output = output[0]["generated_text"][-1]["content"]
                    result_dict = {
                        'image_path': image_path,
                        'caption': prompt,
                        'true_desire': true_desire,
                        'result': output.strip(),
                        'option A': logit[0],
                        'option B': logit[1],
                        'option C': logit[2],
                        'option D': logit[3],
                        'option E': logit[4],
                        'option F': logit[5],
                        'option G': logit[6]
                    }
                    all_results.append(result_dict)
        except Exception as e:
            print(f"Error processing batch: {e}")
            traceback.print_exc()
            sys.exit(1)
        et = time.time()
        print(f"Inference completed in {et - st} seconds.")

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
