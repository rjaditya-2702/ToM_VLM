from data.prepare_experiment import expt2, expt3, expt4, expt5
from data.datafactory import Data_Gemma
from torch.utils.data import DataLoader

dataset = expt2(None , "/projects/aiwell/code/aditya_ratan/ToM_VLM/VQA/data/msed_processed.csv", None, 'v1')

# print if len(images) == len(prompt)
# Check lengths
image_name_count = len([item['image_name'] for item in dataset])
prompt_count = len([item['prompt'] for item in dataset])

print(f"Length of image_name: {image_name_count}")
print(f"Length of prompt: {prompt_count}")
print(f"Lengths are equal: {image_name_count == prompt_count}")

dataloader_model = Data_Gemma(dataset)
dataloader = DataLoader(
        dataloader_model, 
        batch_size=10, 
        shuffle=False, 
        collate_fn=lambda x: tuple(zip(*x))
    )  # Return list of tuples as is)
# Iterate through the dataloader and check lengths
for batch_idx, batch in enumerate(dataloader):
    # Assuming your dataset returns (image_path, prompt, true_desire)
    # The collate_fn will create: (tuple_of_image_paths, tuple_of_prompts, tuple_of_true_desires)
    image_paths = batch[0]
    prompts = batch[1]
    if len(image_paths) != len(prompts):
        print(f"Batch {batch_idx}:")
        print(f"  Length of image_paths: {len(image_paths)}")
        print(f"  Length of prompts: {len(prompts)}")
        print(f"  Lengths are equal: {len(image_paths) == len(prompts)}")
    
    # Optional: break after checking first few batches
    # if batch_idx >= 2:
    #     break