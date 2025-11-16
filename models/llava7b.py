import transformers
transformers.set_seed(42)
from transformers import pipeline
import re
import torch
from .dataloader import DataSetLoader
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import time
import os
import gc

os.environ['TORCH_COMPILE_UNSUPPORTED']='1'
# Enable deterministic operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 42  # or any integer you prefer
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
       
class LlavaModel13B:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf"):
        self.pipe = pipeline("image-text-to-text", model=model_id, device = 'cuda')        
        self.pipe.tokenizer.padding_side = 'left'
        if self.pipe.tokenizer.pad_token is None:
            self.pipe.tokenizer.pad_token = self.pipe.tokenizer.eos_token
        self.pipe.model = torch.compile(self.pipe.model, mode="reduce-overhead")

    @torch.inference_mode()
    def infer(self, dataloader):
        
        # Storage for results
        all_results = []
        print("Processing with images...")
        st = time.time()
        
        # Run inference for image inputs
        for batch in tqdm(dataloader):
            # Unpack the batch tuple
            image_paths, messages_list, output_texts = batch

            # output = self.pipe(text=messages_list, max_new_tokens=300, do_sample=False)
            # o1 = [out["generated_text"][-1]["content"] for out in output]
            
            batch_outputs = self.pipe(
                text=messages_list,  # List of all messages in the batch
                max_new_tokens=300,
                do_sample=False,
                batch_size=len(messages_list)  # Specify batch size
            )
            # Store results for this batch
            for i, (image_path, messages, true_desire, output) in enumerate(
                zip(image_paths, messages_list, output_texts, batch_outputs)
            ):
                # Extract caption from messages
                caption_text = messages[0]["content"][1]["text"] if len(messages[0]["content"]) > 1 else ""
                output = output[0]["generated_text"][-1]["content"]
                result_dict = {
                    'image_path': image_path,
                    'caption': caption_text,
                    'true_desire': true_desire,
                    'with_image_result': format(output.strip())
                }
                all_results.append(result_dict)
            torch.cuda.empty_cache()
        et = time.time()
        print(f"Inference completed in {et - st} seconds.")

        # Build DataFrame
        df_data = []
        model_name = "llava-1.5-13b-hf"  # You can make this configurable
        
        for result in all_results:
            # Extract image name from path
            image_name = os.path.basename(result['image_path'])
            
            row = [
                version,
                image_name,
                result['caption'],
                model_name,
                result['true_desire'],
                result.get('with_image_result', ''),
                result.get('without_image_result', '')
            ]
            df_data.append(row)
        
        output_df = pd.DataFrame(df_data, columns=df_cols)
        torch.cuda.empty_cache()
        return output_df
    
