from torch.utils.data import Dataset
import os
import pandas as pd
import copy
from typing import List, Dict, Any
from PIL import Image

class Data_Qwen(Dataset):
    def __init__(self, data):
        self.data = data
        self.message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "file://{}"
                        },
                        {
                            "type": "text",
                            "text": "{}"
                        }
                    ]
                }
            ]
        self.no_img_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "{}"
                        }
                    ]
                }
            ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = "/projects/aiwell/code/aditya_ratan/ToM_VLM/existing_datasets/msed/dev/images"
        row = self.data[idx]
        
        true_label = row['true_desire']
        prompt = row['prompt']
        image_name = row['image_name']

        if image_name is None:
            template = copy.deepcopy(self.no_img_message)
            template[0]["content"][0]["text"] = prompt
            image_path = None
        else:
            image_path = os.path.join(image_path, image_name)
            template = copy.deepcopy(self.message)
            template[0]["content"][0]["image"] = f"file://{image_path}"
            template[0]["content"][1]["text"] = prompt

        return image_path, prompt, template, true_label

class Data_Llama:
    def __init__(self, data):
        self.message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image"
                        },
                        {
                            "type": "text",
                            "text": "{}"
                        }
                    ]
                }
            ]
        
        self.no_img_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "{}"
                        }
                    ]
                }
            ]
        
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = "/projects/aiwell/code/aditya_ratan/ToM_VLM/existing_datasets/msed/dev/images"
        row = self.data[idx]
        
        true_label = row['true_desire']
        prompt = row['prompt']        
        image_name = row['image_name']

        if image_name is None:
            image_path = None
            template = copy.deepcopy(self.no_img_message)
            template[0]["content"][0]["text"] = prompt
        else:
            image_path = os.path.join(image_path, image_name)
            template = copy.deepcopy(self.message)
            template[0]["content"][1]["text"] = prompt

        return image_path, prompt, template, true_label

class Data_Llava7B:
    def __init__(self, data):
        self.message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "{}"
                        },
                        {
                            "type": "text",
                            "text": "{}"
                        }
                    ]
                }
            ]
        self.no_img_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "{}"
                        }
                    ]
                }
            ]
        
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = "/projects/aiwell/code/aditya_ratan/ToM_VLM/existing_datasets/msed/dev/images"
        row = self.data[idx]
        
        true_label = row['true_desire']
        prompt = row['prompt']
        image_name = row['image_name']

        if image_name is None:
            image_path = None
            template = copy.deepcopy(self.no_img_message)
            template[0]["content"][0]["text"] = prompt
        else:
            image_path = os.path.join(image_path, image_name)
            template = copy.deepcopy(self.message)
            template[0]["content"][0]["image"] = image_path
            template[0]["content"][1]["text"] = prompt

        return image_path, prompt, template, true_label
        
class Data_Llava13B:
    def __init__(self, data):
        self.message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "{}"
                        },
                        {
                            "type": "text",
                            "text": "{}"
                        }
                    ]
                }
            ]
        self.no_img_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "{}"
                        }
                    ]
                }
            ]
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = "/projects/aiwell/code/aditya_ratan/ToM_VLM/existing_datasets/msed/dev/images"
        row = self.data[idx]
        
        true_label = row['true_desire']
        prompt = row['prompt']
        image_name = row['image_name']
        
        if image_name is None:
            image_path = None
            template = copy.deepcopy(self.no_img_message)
            template[0]["content"][0]["text"] = prompt
        else:
            image_path = os.path.join(image_path, image_name)
            template = copy.deepcopy(self.message)
            template[0]["content"][0]["image"] = image_path
            template[0]["content"][1]["text"] = prompt

        return image_path, prompt, template, true_label

class Data_Gemma:
    def __init__(self, data):
        self.message = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": ""
                        },
                        {
                            "type": "text",
                            "text": ""
                        }
                    ]
                },

            ]
        
        self.no_img_message = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ""
                        }
                    ]
                }
            ]
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = "/projects/aiwell/code/aditya_ratan/ToM_VLM/existing_datasets/msed/dev/images"
        row = self.data[idx]
        
        true_label = row['true_desire']
        prompt = row['prompt']
        image_name = row['image_name']

        if image_name is None:
            image_path = None
            template = copy.deepcopy(self.no_img_message)
            template[1]["content"][0]["text"] = prompt
        else:
            image_path = os.path.join(image_path, image_name)
            template = copy.deepcopy(self.message)
            template[1]["content"][0]["image"] = Image.open(image_path)
            template[1]["content"][1]["text"] = prompt

        return image_path, prompt, template, true_label

class Data_Bagel:
    def __init__(self, data):
        self.message = None
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = "/projects/aiwell/code/aditya_ratan/ToM_VLM/existing_datasets/msed/dev/images"
        row = self.data[idx]
        
        true_label = row['true_desire']
        
        prompt = row['prompt']
        
        image_name = row['image_name']

        if image_name is None:
            image_path = None
        else:
            image_path = os.path.join(image_path, image_name)

        return image_path, prompt, true_label

class DataFactory:
    # MODEL_CLASSES = {
    #     "qwen": Data_Qwen,
    #     "llama": Data_Llama,     
    #     "llava7b": Data_Llava7B,
    #     "llava13b": Data_Llava13B,
    #     "gemma": Data_Gemma,
    #     "bagel": Data_Bagel,
    # }
    
    @staticmethod
    def create_model(model_name, data):
        MODEL_CLASSES = {
            "qwen": Data_Qwen,
            "llama": Data_Llama,     
            "llava7b": Data_Llava7B,
            "llava13b": Data_Llava13B,
            "gemma": Data_Gemma,
            "bagel": Data_Bagel,
        }
        model_class = MODEL_CLASSES[model_name]
        return model_class(data)