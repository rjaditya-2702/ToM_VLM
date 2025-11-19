import transformers
transformers.set_seed(42)
from transformers import pipeline
import torch
import os
import re
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

import numpy as np
import traceback

import warnings
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