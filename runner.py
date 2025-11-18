# inference/runner.py

import pandas as pd
import os
import sys

from models.modelfactory import ModelFactory
from data.datafactory import DataFactory

from data.prepare_experiment import expt1
from data.prepare_experiment import expt2
from data.prepare_experiment import expt3
from data.prepare_experiment import expt4

from torch.utils.data import DataLoader

CUR_DIR = os.path.dirname(__file__) # VQA
SAVE_DIR = os.path.join(CUR_DIR , "results")

def run_benchmark(
            dataset = 'msed',
            model_name= None,
            experiment_type= None,
            batch_size= 16,
            n = None
        ):
    
    if model_name == None:
        raise "Model cannot be None"

    if dataset == 'msed':
        image_path = '/projects/aiwell/code/aditya_ratan/ToM_VLM/existing_datasets/msed/dev/images'
    else:
        image_path = None
    
    if image_path == None:
        raise "Please pass the correct dataset name"

    data = None
    match experiment_type:
        case 'expt1':
            data = expt1(dataset, "/projects/aiwell/code/aditya_ratan/ToM_VLM/VQA/data/msed_processed.csv", n)
        case 'exp2':
            data = expt2(dataset, "/projects/aiwell/code/aditya_ratan/ToM_VLM/VQA/data/msed_processed.csv", n)
        case 'expt3':
            data = expt3(dataset, "/projects/aiwell/code/aditya_ratan/ToM_VLM/VQA/data/msed_processed.csv", n)
        case 'expt4':
            data = expt4(dataset, "/projects/aiwell/code/aditya_ratan/ToM_VLM/VQA/data/msed_processed.csv", n)
    if data == None:
        raise "define a proper experiment please :)"

    # data: List[dict]
    # data[i] = {
    #         'image_name': "black_image.jpg",  
    #         'prompt': PROMPT.format(addition),
    #         'true_desire': row['Desire']
    #     }

    dataloader_model = DataFactory.create_model(model_name, data)
    dataloader = DataLoader(
        dataloader_model, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=3,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True,  # Pin memory for faster GPU transfer
        prefetch_factor=2  # Number of batches to prefetch per worker
        )  # Return list of tuples as is)

    model = ModelFactory.create_model(model_name)
    results = model.infer(dataloader) # output_df = pd.DataFrame(df_data, columns=['ImageName', 'Model', 'True Label', 'Predicted Label', 'Output Text', 'curiosity', 'family', 'tranquility', 'vengeance', 'social-contact', 'romance', 'none'])

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(SAVE_DIR, f"{model_name}_{experiment_type}.csv")
    results.to_csv(SAVE_PATH)