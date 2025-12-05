import transformers
import torch
import os
import re
import time
import pandas as pd
import sys
import numpy as np
import traceback
import warnings

from typing import List, Dict, Any
from transformers import LlavaForConditionalGeneration, AutoProcessor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

transformers.set_seed(42)
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

class Llava13BModel:
    def __init__(self, model_id="llava-hf/llava-1.5-13b-hf"):        
        safe_cuda_init()
        
        self.model_name = "llava-1.5-13b-hf"
        
        # Load processor and model directly
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.model.eval()

        # Define target token IDs for options A-G
        self.target_token_ids = [
            self.processor.tokenizer.encode("A", add_special_tokens=False)[0],
            self.processor.tokenizer.encode("B", add_special_tokens=False)[0],
            self.processor.tokenizer.encode("C", add_special_tokens=False)[0],
            self.processor.tokenizer.encode("D", add_special_tokens=False)[0],
            self.processor.tokenizer.encode("E", add_special_tokens=False)[0],
            self.processor.tokenizer.encode("F", add_special_tokens=False)[0],
            self.processor.tokenizer.encode("G", add_special_tokens=False)[0],
        ]

        # self.pipe.model = torch.compile(self.pipe.model, mode="reduce-overhead")

    @torch.inference_mode()
    def _process_batch(self, templates: List[dict], prompts:List[Any], max_new_tokens: int):
        """
        Process a single batch: generate text and extract first token logits
        
        Args:
            templates: List of message templates (batch)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            batch_outputs: List of outputs in format [{"generated_text": [...]}]
            first_token_logits: List of (vocab,) tensors for first generated token
        """

        images = []
        for template in templates:
            # Extract image if present
            image = None
            # text_parts = []
            
            for message in template:
                if message["role"] == "user":
                    for content in message["content"]:
                        if content["type"] == "image" and content["url"]:
                            image = Image.open(content["url"])

            images.append(image)

        if len(images) == 0:
            images = [None]*len(templates)

        formatted_texts = self.processor.apply_chat_template(
            templates,
            tokenize=False,
        )

        # Process each sample individually, then manually batch
        # This is more reliable for multimodal batching
        all_input_ids = []
        all_attention_masks = []
        all_pixel_values = []
        
        for text, img in zip(formatted_texts, images):
            single_input = self.processor(
                text=text,
                images=img,  # Single image
                return_tensors="pt",
            )
            all_input_ids.append(single_input['input_ids'])
            all_attention_masks.append(single_input['attention_mask'])
            if 'pixel_values' in single_input:
                all_pixel_values.append(single_input['pixel_values'])
        
        # Pad and stack
        
        # Pad input_ids (need to squeeze and re-add batch dim)
        input_ids_list = [x.squeeze(0) for x in all_input_ids]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id, padding_side='left')
        
        attention_mask_list = [x.squeeze(0) for x in all_attention_masks]
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        
        inputs = {
            'input_ids': input_ids.to('cuda'),
            'attention_mask': attention_mask.to('cuda'),
        }
        
        if all_pixel_values:
            # Stack pixel values - they should all be same shape
            pixel_values = torch.cat(all_pixel_values, dim=0).to('cuda')
            inputs['pixel_values'] = pixel_values
        
        # Generate with output logits
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False  # Deterministic generation
            )
        
        # Extract generated sequences
        generated_ids = outputs.sequences
        
        # Decode outputs - remove input tokens
        input_length = inputs['input_ids'].shape[1]
        batch_outputs = []
        
        for i, gen_id in enumerate(generated_ids):
            # Only decode the newly generated tokens
            generated_tokens = gen_id[input_length:]
            generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
            
            # Format output to match expected structure
            output_entry = [{
                "generated_text": [
                    *templates[i],  # Include input messages
                    {"role": "assistant", "content": generated_text}
                ]
            }]
            batch_outputs.append(output_entry)
        
        # Extract first token logits
        # outputs.scores is a tuple of (vocab_size,) tensors for each generation step
        # The first element corresponds to the first generated token
        if len(outputs.scores) > 0:
            first_token_logits = outputs.scores[0]  # (batch_size, vocab_size)
            
            # Convert to list of individual (vocab,) tensors on CPU
            first_token_logits_list = [logits.cpu().float() for logits in first_token_logits]
        else:
            raise Exception("1st generated token output logits are of length 0????????")
        
        return batch_outputs, first_token_logits_list

    @torch.inference_mode()
    def infer(self, dataloader):
        
        # Storage for results
        all_results = []
        print("Processing with images...")
        st = time.time()
        
        # Run inference for image inputs
        try:
            for batch in tqdm(dataloader):
                image_paths, prompts, templates, true_labels = batch
                batch_outputs, first_token_logits = self._process_batch(
                    templates=list(templates),
                    prompts=list(prompts),
                    max_new_tokens=300
                )
                option_probs = []  # list of (7,) tensors
                
                for raw_logit in first_token_logits:  # each is (vocab,)
                    probs = torch.nn.functional.softmax(raw_logit, dim=-1)  # (vocab,)
                    position_probs = probs[self.target_token_ids]  # (7,)
                    option_probs.append(position_probs)
                
                # Store results for this batch
                for i, (image_path, prompt, true_desire, output, logit) in enumerate(
                    zip(image_paths, prompts, true_labels, batch_outputs, option_probs)
                ):
                    # Extract caption from messages
                    output_text = output[0]["generated_text"][-1]["content"]
                    
                    result_dict = {
                        'image_path': image_path,
                        'caption': prompt,
                        'true_desire': true_desire,
                        'result': output_text.strip(),
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
            print(f"Error running inference - {e}")
            traceback.print_exc()
            sys.exit(1)
        et = time.time()
        print(f"Inference completed in {et - st} seconds.")

        # Build DataFrame
        df_data = []
        
        for result in all_results:
            # Extract image name from path
            p = result['image_path']
            if p:
                image_name = os.path.basename(p)
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
