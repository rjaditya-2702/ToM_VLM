from .models.qwen import QwenModel
from .models.llama import LlamaModel
from .models.llava7b import Llava7BModel
from .models.llava13b import Llava13BModel
from .models.gemma import GemmaModel
from .models.bagel import BagelModel

from transformers import pipeline

class AbstractModel:
    def __init__(self, model_id):
        self.pipe = pipeline("image-text-to-text", model_id, device_map='auto')

        # Set left padding for decoder-only model
        self.pipe.tokenizer.padding_side = 'left'

        # Set pad token if not set
        if self.pipe.tokenizer.pad_token is None:
            self.pipe.tokenizer.pad_token = self.pipe.tokenizer.eos_token
        
        self.pipe.model = torch.compile(self.pipe.model, mode="reduce-overhead")
        

class ModelFactory:
    MODEL_CLASSES = {
        "qwen": QwenModel,
        "llama": LlamaModel,     
        "llava7b": Llava7BModel,
        "llava13b": Llava13BModel,
        "gemma": GemmaModel,
        "bagel": BagelModel,
    }

    @staticmethod
    def create_model(model_name):
        if model_name not in self.MODEL_CLASSES:
            raise ValueError(f"Model name not in list - {list(self.MODEL_CLASSES.keys())}")

        model_class = self.MODEL_CLASSES[model_name]
        return model_class()
