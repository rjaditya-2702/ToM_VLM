import transformers
from transformers import pipeline
import torch
import os
import re
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
import traceback

import warnings
warnings.filterwarnings("ignore")

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from huggingface_hub import snapshot_download
from .Bagel.modeling.bagel.bagel import Bagel, BagelConfig
from .Bagel.modeling.bagel.qwen2_navit import Qwen2Config, Qwen2ForCausalLM
from .Bagel.modeling.bagel.siglip_navit import SiglipVisionConfig, SiglipVisionModel
from .Bagel.modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer

from .Bagel.data.data_utils import add_special_tokens
from .Bagel.modeling.bagel.qwen2_navit import NaiveCache

from .Bagel.inferencer import InterleaveInferencer

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

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CUR_DIR = os.path.dirname(__file__)

class BagelModel:
    def __init__(self):
        # Step 0: clone Bagel REPO in models/ folders 
        # (else imports will fail)
        # 
        # manual step ;-;

        # Step 1 : Download snapshot
        self.save_dir = os.path.join(CUR_DIR, 'bagel_parameters')
        repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
        cache_dir = os.path.join(self.save_dir, "cache")

        snapshot_download(cache_dir=cache_dir,
            local_dir=self.save_dir,
            repo_id=repo_id,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
        )

        # Step 2: Initialization
        llm_config = Qwen2Config.from_json_file(os.path.join(self.save_dir, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(self.save_dir, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        self.model_config = BagelConfig(
            visual_gen=False, 
            visual_und=True, 
            llm_config=llm_config,
            vit_config=vit_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        ) # avoiding VAE because it is used for generation only.

        with init_empty_weights():
            self.language_model = Qwen2ForCausalLM(llm_config)
            self.vit_model = SiglipVisionModel(vit_config)
            self.model = Bagel(
                self.language_model, 
                self.vit_model, 
                self.model_config, 
                torch_dtype = torch.bfloat16 #compromise on precision :(
            )
        
        # Tokenizer Preparing
        self.tokenizer = Qwen2Tokenizer.from_pretrained(self.save_dir)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)
        self.vit_transform = ImageTransform(980, 224, 14)
    
    @torch.inference_mode()
    def generate_text_custom(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):

        curr_tokens = packed_start_tokens

        packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
        query_lens = torch.ones_like(curr_tokens)
        packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
            0, len(key_values_lens), 
            device=key_values_lens.device, 
            dtype=key_values_lens.dtype
        )

        uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
        for i in range(len(uppacked)):
            uppacked[i] += i
        packed_key_value_indexes = torch.cat(uppacked, dim=0)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values
        packed_query_sequence = output.packed_query_sequence
        pred_logits = self.language_model.lm_head(packed_query_sequence)
        return pred_logits

    @torch.inference_mode()
    def get_logits(self, prompts, images):
        device = next(self.model.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        outputs = []

        # add image
        for prompt, image_path in zip(prompts, images):
            image = Image.open(image_path)
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

            # add text
            generation_input, newlens, new_rope = self.prepare_prompts(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                prompts=[prompt],
                tokenizer=tokenizer, 
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

            # decode
            generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            
            logits = self.generate_text_custom(past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,)
            outputs.append(logits.cpu().float())
            
        return outputs
            

    @torch.inference_mode()
    def infer(self, dataloader):

        # prepare GPU and load model - 
        max_mem_per_gpu = "40GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        print(device_map)

        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        # Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
        self.model = load_checkpoint_and_dispatch(
            self.model,
            checkpoint=os.path.join(self.save_dir, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )

        self.model = self.model.eval()
        print('Model loaded')

        inferencer = InterleaveInference(
            model = self.model,
            vae_model = None,
            tokenizer = self.tokenizer,
            vae_transform = None,
            vit_transformer = self.vit_transform,
            new_token_ids = self.new_token_ids
        )

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # Storage for results
        all_results = []
        print("Processing with images...")
        st = time.time()
        
        # Run inference for image inputs
        try:
            for batch in tqdm(dataloader):
                # Unpack the batch tuple
                image_paths, prompts, templates, true_labels = batch

                inference_hyper=dict(
                    max_think_token_n=1000,
                    do_sample=False,
                    # text_temperature=0.3,
                )
                batch_outputs = []
                for img_path, prompt in zip(img_paths, prompts):
                    output_d = inferencer(image = Image.open(img_path), text = prompt, understanding_output=True, **inference_hyper)
                    batch_outputs.append(output_d['text'])

                # do it again, but now, we need logits :)
                with torch.no_grad():
                    logit_outputs = self.get_logits(prompts, images)
                                    
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
                    output = None # get the generated text
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