from .qwen import QwenModel
from .llama import LlamaModel
from .llava7b import Llava7BModel
from .llava13b import Llava13BModel
from .gemma import GemmaModel
from .bagel import BagelModel

from transformers import pipeline

class ModelFactory:

    @staticmethod
    def create_model(model_name):
        MODEL_CLASSES = {
            "qwen": QwenModel,
            "llama": LlamaModel,     
            "llava7b": Llava7BModel,
            "llava13b": Llava13BModel,
            "gemma": GemmaModel,
            "bagel": BagelModel,
        }
        if model_name not in MODEL_CLASSES:
            raise ValueError(f"Model name not in list - {list(MODEL_CLASSES.keys())}")

        model_class = MODEL_CLASSES[model_name]
        return model_class()