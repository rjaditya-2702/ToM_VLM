import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm

class BagelModel:
    def __init__(self, model_id="ByteDance-Seed/BAGEL-7B-MoT"):
        """
        Initialize BAGEL model from ByteDance
        BAGEL uses Mixture-of-Transformer-Experts (MoT) architecture
        Note: Requires trust_remote_code=True as it's not yet integrated into transformers
        """
        print("Loading BAGEL model... This may take a while.")
        
        # Load processor first
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Load model with trust_remote_code
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Set padding
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = 'left'
        
        # Try to compile (may not work with custom architecture)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("Model compiled successfully")
        except Exception as e:
            print(f"Model compilation not supported: {e}")
            print("Continuing without compilation...")

    @torch.inference_mode()
    def infer(self, dataloader):
        """
        Run inference on a given dataset
        BAGEL supports multimodal understanding through its MoT architecture
        """
        # Storage for results
        all_results = []
        print("Processing with BAGEL...")

        st = time.time()
        
        # Run inference - process individually for stability
        for batch in tqdm(dataloader):
            image_paths, prompts, templates, true_labels = batch
            
            # Process each sample individually
            for image_path, prompt, template, true_label in zip(image_paths, prompts, templates, true_labels):
                try:
                    # Apply chat template
                    text = self.processor.apply_chat_template(
                        template,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    # Process inputs
                    inputs = self.processor(
                        text=[text],
                        return_tensors="pt"
                    ).to(self.model.device)

                    # Generate
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=300,
                        do_sample=False
                    )

                    # Decode output
                    generated_ids_trimmed = generated_ids[0][inputs['input_ids'].shape[1]:]
                    output = self.processor.decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    # Get logits by doing a forward pass
                    with torch.no_grad():
                        logit_outputs = self.model(**inputs, return_dict=True)

                    # The 'logits' tensor has shape (batch_size, sequence_length, vocab_size)
                    raw_logits = logit_outputs.logits
                    
                    # Get token IDs for options A-G
                    target_tokens = ["A", "B", "C", "D", "E", "F", "G"]
                    target_token_ids = self.processor.tokenizer.convert_tokens_to_ids(target_tokens)
                    
                    # Get the logits for the first generated token (right after input)
                    first_token_logits = raw_logits[0, inputs['input_ids'].shape[1] - 1]
                    probs = torch.nn.functional.softmax(first_token_logits, dim=-1)
                    position_probs = probs[target_token_ids].cpu().numpy()

                    # Store result
                    result_dict = {
                        'image_path': image_path,
                        'caption': prompt,
                        'true_desire': true_label,
                        'result': output.strip(),
                        'option A': float(position_probs[0]),
                        'option B': float(position_probs[1]),
                        'option C': float(position_probs[2]),
                        'option D': float(position_probs[3]),
                        'option E': float(position_probs[4]),
                        'option F': float(position_probs[5]),
                        'option G': float(position_probs[6])
                    }
                    all_results.append(result_dict)

                except Exception as e:
                    print(f"Error processing sample {image_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        et = time.time()
        print(f"Processing time: {et - st:.2f} seconds")

        # Build DataFrame
        df_data = []
        model_name = "BAGEL-7B-MoT"

        # Assuming GOLD is defined somewhere
        GOLD = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

        for result in all_results:
            # Extract image name from path
            image_name = os.path.basename(result['image_path'])
            predictions = [
                result['option A'],
                result['option B'],
                result['option C'],
                result['option D'],
                result['option E'],
                result['option F'],
                result['option G']
            ]
            model_prediction = GOLD[int(np.argmax(predictions))]

            row = [
                image_name,
                result['caption'],
                model_name,
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
                'ImageName', 'Caption', 'Model', 'True Label', 'Predicted Label', 
                'Output Text', 'curiosity', 'family', 'tranquility', 'vengeance', 
                'social-contact', 'romance', 'none'
            ]
        )
        return output_df