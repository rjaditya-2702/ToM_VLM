# inference/runner.py

import pandas as pd
import os
import sys

CUR_DIR = os.path.dirname(__file__) # inference
PARENT_DIR = os.path.dirname(CUR_DIR) # VQA
SAVE_DIR = os.path.join(PARENT_DIR , "results")
# sys.path.insert(0, PARENT_DIR)

from VQA.models.modelfactory import ModelFactory
from VQA.data.datafactory import DataFactory
from torch.utils.data import DataLoader
from VQA.data.prepare_experiment import expt1
from VQA.data.prepare_experiment import expt2
from VQA.data.prepare_experiment import expt3
from VQA.data.prepare_experiment import expt4
from VQA.data.prepare_experiment import expt5



def run_benchmark(
            dataset = 'msed',
            model_name= None,
            experiments= None,
            batch_size= 16,
            n = None,
            v = None
        ):
    
    if model_name == None:
        raise ValueError("Model cannot be None")
    model = ModelFactory.create_model(model_name)
    print(f"Created a model instance for {model_name}")

    if dataset == 'msed':
        image_path = '/projects/aiwell/code/aditya_ratan/ToM_VLM/existing_datasets/msed/dev/images'
    else:
        image_path = None
    
    if image_path == None:
        raise ValueError("Please pass the correct dataset name")

    print("Model name and dataset validated")

    for experiment_type in experiments:

        print("="*80)
        print(f'\t\t\tRunning {experiment_type} for model {model_name}')
        print("="*80)
        print()

        data = None
        
        match experiment_type:
            case 'expt1':
                data = expt1(dataset, "/projects/aiwell/code/aditya_ratan/ToM_VLM/VQA/data/msed_processed.csv", n, v)
            case 'expt2':
                data = expt2(dataset, "/projects/aiwell/code/aditya_ratan/ToM_VLM/VQA/data/msed_processed.csv", n, v)
            case 'expt3':
                data = expt3(dataset, "/projects/aiwell/code/aditya_ratan/ToM_VLM/VQA/data/msed_processed.csv", n, v)
            case 'expt4':
                data = expt4(dataset, "/projects/aiwell/code/aditya_ratan/ToM_VLM/VQA/data/msed_processed.csv", n, v)
            case 'expt5':
                data = expt5(dataset, "/projects/aiwell/code/aditya_ratan/ToM_VLM/VQA/data/msed_processed.csv", n, v)
        if data is None:
            raise Exception("define a proper experiment please :)")
        
        print("Data prepared")

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
        
        print("Converted data to a torch dataloader")

        results = model.infer(dataloader) # output_df = pd.DataFrame(df_data, columns=['ImageName', 'Model', 'True Label', 'Predicted Label', 'Output Text', 'curiosity', 'family', 'tranquility', 'vengeance', 'social-contact', 'romance', 'none'])

        print("Results fetched")

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR, exist_ok=True)
        SAVE_PATH = os.path.join(SAVE_DIR, f"{model_name}_{experiment_type}_{v}.csv")
        results.to_csv(SAVE_PATH)

        print("Results saved to {0}".format(SAVE_PATH))
    del model