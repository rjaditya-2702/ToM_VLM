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

os.environ['TORCH_COMPILE_UNSUPPORTED']='1'
# Enable deterministic operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 42  # or any integer you prefer
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class LLamaModel:
    def __init__(self, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
        self.model_name = "Llama-3.2-11B-Vision-Instruct"  # You can make this configurable
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            dtype = torch.bfloat16,
            device_map = 'cuda'
        )
        self.model = torch.compile(self.model, mode="reduce-overhead")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = 'left'
        
    
    @torch.inference_mode()
    def infer(self, dataloder):
        
        # Storage for results
        all_results = []
        
        print("Processing with images...")
        st = time.time()
        
        # Run inference for image inputs
        for batch in tqdm(dataloader_images):
            # Unpack the batch tuple
            image_paths, messages_list, output_texts = batch
            images = [[Image.open(p)] for p in image_paths]
            
            try:
                # Process all messages and images in the batch
                # batch_texts = []
                # batch_images = []
                
                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages_list,
                    tokenizer=False, 
                    add_generation_prompt=True
                )
                    
                # batch_texts.append(text)
                # batch_images.append(image_inputs)
                
                # Prepare all inputs at once
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
                
                # Store results for this batch
                for i, (image_path, messages, true_desire, output) in enumerate(
                    zip(image_paths, messages_list, output_texts, outputs)
                ):
                    # Extract caption from messages
                    caption_text = messages[0]["content"][1]["text"] if len(messages[0]["content"]) > 1 else ""
                    
                    result_dict = {
                        'image_path': image_path,
                        'caption': caption_text,
                        'true_desire': true_desire,
                        'with_image_result': format(output.strip())
                    }
                    all_results.append(result_dict)
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                sys.exit(1)
                # Fall back to sequential processing for this batch if batch processing fails
                for image_path, messages, true_desire in zip(image_paths, messages_list, output_texts):
                    caption_text = messages[0]["content"][1]["text"] if len(messages[0]["content"]) > 1 else ""
                    result_dict = {
                        'image_path': image_path,
                        'caption': caption_text,
                        'true_desire': true_desire,
                        'with_image_result': "ERROR"
                    }
                    all_results.append(result_dict)
        
        et = time.time()
        print(f"Image processing time: {et - st:.2f} seconds")
        
        # Build DataFrame
        df_data = []
        
        for result in all_results:
            # Extract image name from path
            image_name = os.path.basename(result['image_path'])
            
            row = [
                version,
                image_name,
                result['caption'],
                self.model_name,
                result['true_desire'],
                result.get('with_image_result', ''),
                result.get('without_image_result', '')
            ]
            df_data.append(row)
        
        output_df = pd.DataFrame(df_data, columns=df_cols)
        
        return output_df