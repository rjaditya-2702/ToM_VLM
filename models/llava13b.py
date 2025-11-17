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
       
class Llava13BModel:
    def __init__(self, model_id="llava-hf/llava-1.5-13b-hf"):
        self.pipe = pipeline("image-text-to-text", model=model_id, device = 'cuda')        
        self.pipe.tokenizer.padding_side = 'left'
        if self.pipe.tokenizer.pad_token is None:
            self.pipe.tokenizer.pad_token = self.pipe.tokenizer.eos_token
        self.pipe.model = torch.compile(self.pipe.model, mode="reduce-overhead")

        self.model_name = "llava-hf/llava-1.5-13b-hf"

    @torch.inference_mode()
    def infer(self, dataloader):
        
        # Storage for results
        all_results = []
        print("Processing with images...")
        st = time.time()
        
        # Run inference for image inputs
        for batch in tqdm(dataloader_images):
            # Unpack the batch tuple
            image_paths, prompts, templates, true_labels = batch

            batch_outputs = self.pipe(
                text=templates,  # List of all messages in the batch
                max_new_tokens=300,
                do_sample=False,
                batch_size=len(messages_list)  # Specify batch size
            )

            # do it again, but now, we need logits :)
            with torch.no_grad():
                logit_outputs = self.pipe.model(**inputs, output_hidden_states=False, return_dict=True)

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
        
        output_df = pd.DataFrame(df_data, columns=['ImageName', 'Model', 'True Label', 'Predicted Label', 'Output Text', 'curiosity', 'family', 'tranquility', 'vengeance', 'social-contact', 'romance', 'none'])
        return output_df    
  
