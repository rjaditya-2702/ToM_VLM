import torch
import numpy as np
import pandas as pd
import os

from copy import deepcopy

PROMPT = """
The following are multiple choice questions (with exactly one answer).

{}
A. curiosity - desire to discover, learn, explore, understand
B. family - desire for family connection, relatives, family bonds
C. tranquility - desire for peace, calm, rest, relaxation
D. vengeance - desire for revenge, retribution, payback
E. social-contact
F. romance - desire for romantic love, partnership, intimate relationships
G. none - doesn't fit any desire option above
Answer:
"""

random_seed = 42 

def msed_dev_dataset_prep(csv_path: str = "/projects/aiwell/code/aditya_ratan/ToM_VLM/existing_datasets/msed/dev/dev.csv", 
                            image_dir: str = "/projects/aiwell/code/aditya_ratan/ToM_VLM/existing_datasets/msed/dev/images", 
                            output_path: str = "/projects/aiwell/code/aditya_ratan/ToM_VLM/VQA/data/msed_processed.csv"):
    """
    Revised dataset from the dev/ path
    """
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    # os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path) # Read the CSV file

    images_names = [] # Copy corresponding images
    
    for idx, _ in df.iterrows():
        image_name = f"{idx+1}.jpg" # Image filename is {row_index + 1}.jpg
        images_names.append(image_name)
        
    df['ImageName'] = images_names
    
    # sampled_csv_path = os.path.join(output_dir, 'sample.csv') # Save sampled CSV
    df.to_csv(output_path, index=False)
    print(f"Data CSV saved to: {output_path}")


def expt1(data_name, dataset_path, n = None):
    """
    if experiment == "expt1": use blank image, no question
    """
    if not os.path.exists(dataset_path):
        if data_name == 'msed':
            msed_dev_dataset_prep(output_path = dataset_path)
    df = pd.read_csv(dataset_path)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=random_seed).reset_index(drop=True)
    else:
        n = len(df)
    dataset = []
    for i in range(n):
        dataset.append({
            'image_name': "black_image.jpg",
            'prompt': deepcopy(PROMPT).format(''),
            'true_desire': None  # No ground truth for blank images
        })
    
    return dataset
    

def expt2(data_name, dataset_path, n = None):
    """
    if experiment == "expt2": use blank image, with question
    """
    if not os.path.exists(dataset_path):
        if data_name == 'msed':
            msed_dev_dataset_prep(output_path = dataset_path)
    df = pd.read_csv(dataset_path)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=random_seed).reset_index(drop=True)
    
    dataset = []
    for idx, row in df.iterrows():
        title = row['Title']
        caption = row['Caption']
        addition = f"Title: {title}\nCaption: {caption}\n"
        dataset.append({
            'image_name': "black_image.jpg",  
            'prompt': deepcopy(PROMPT).format(addition),
            'true_desire': row['Desire']
        })
    
    return dataset


def expt3(data_name, dataset_path, n = None):
    """
    if experiment == "expt3": use real image, no question
    """
    if not os.path.exists(dataset_path):
        if data_name == 'msed':
            msed_dev_dataset_prep(output_path = dataset_path)
    df = pd.read_csv(dataset_pat2)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=random_seed).reset_index(drop=True)
    
    dataset = []
    for idx, row in df.iterrows():
        dataset.append({
            'image_name': row['ImageName'],  
            'prompt': deepcopy(PROMPT).format(''),
            'true_desire': row['Desire']
        })
    
    return dataset

def expt4(data_name, dataset_path, n = None):
    """
    if experiment == "expt4": use real image, with question
    """
    if not os.path.exists(dataset_path):
        if data_name == 'msed':
            msed_dev_dataset_prep(output_path = dataset_path)
    df = pd.read_csv(dataset_path)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=random_seed).reset_index(drop=True)
    
    dataset = []
    for idx, row in df.iterrows():
        title = row['Title']
        caption = row['Caption']
        addition = f"Title: {title}\nCaption: {caption}\n"
        dataset.append({
            'image_name': row['ImageName'],  
            'prompt': deepcopy(PROMPT).format(addition),
            'true_desire': row['Desire']
        })
    
    return dataset

def expt5(data_name, dataset_path, n = None):
    # No image and caption
    pass