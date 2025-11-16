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

os.environ['TORCH_COMPILE_UNSUPPORTED']='1'
# Enable deterministic operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 42  
transformers.set_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

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

class QwenModel:
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct"):
        self.model_name = "Qwen3-VL-8B-Instruct"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype = torch.bfloat16, device_map="auto" ) #, attn_implementation="flash_attention_2")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = 'left'
        self.model = torch.compile(self.model, mode="reduce-overhead")

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
                    do_sample=False
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
        print(f"Processing time: {et - st:.2f} seconds")
        
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
