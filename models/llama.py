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

GOLD = {
    0:'curiosity',
    1:'family',
    2:'tranquility',
    3:'vengeance',
    4:'social-contact',
    5:'romance',
    6:'none'
}

class LlamaModel:
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
            image_paths, prompts, tempaltes, true_labels = batch
            images = [[Image.open(p)] for p in image_paths]
            
            try:                
                # Prepare all inputs at once
                inputs = self.processor(
                    text=templates,
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
                    logit_outputs = self.model(**inputs, output_hidden_states=False, return_dict=True)

                # The 'logits' tensor has shape (batch_size, sequence_length, vocab_size)
                raw_logits = logit_outputs.logits # (batch_size, sequence_length, vocab_size)
                target_tokens = ["A", "B", "C", "D", "E", "F", "G"]
                target_token_ids = self.processor.tokenizer.convert_tokens_to_ids(target_tokens)
                option_probs = [] # bs x 7
                for raw_logit, in_ids in zip(raw_logits, inputs.input_ids): #(seq_, v)
                    first_token = raw_logit[len(in_ids)] # (1,v)
                    probs = torch.nn.functional.softmax(first_token, dim=-1) #(1,v)
                    position_probs = probs[token_ids]
                
                # Store results for this batch
                for i, (image_path, prompt, true_desire, output, logit) in enumerate(
                    zip(image_paths, prompts, true_labels, outputs, option_probs)
                ):
                    
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
        
        output_df = pd.DataFrame(df_data, columns=['ImageName', 'Model', 'True Label', 'Predicted Label', 'Output Text', 'curiosity', 'family', 'tranquility', 'vengeance', 'social-contact', 'romance', 'none'])
        return output_df