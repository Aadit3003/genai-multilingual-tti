""" Visualization script to generate an ImageGrid, with the prompts on the Y-axis, and the timesteps (checkpoints) on the X-axis


Run this only after running the training scripts (teacher_learning.py and train_text_to_image_lora_rks.py) which stores
the checkpoints needed to visualize the results """

import torch
import os
from huggingface_hub import model_info
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch_type = torch.bfloat16
assert torch.cuda.is_available(), "Torch Cuda is NOT available!"

def visualize_results_with_imagegrid(results, prompts, checkpoints):
    """
    Visualizes the generated images in a grid layout using ImageGrid.
    """
    num_prompts = len(prompts)
    num_checkpoints = len(checkpoints)

    fig = plt.figure(figsize=(num_checkpoints * 3, num_prompts * 3))
    grid = ImageGrid(fig, 111, 
                     nrows_ncols=(num_prompts, num_checkpoints), 
                     axes_pad=0.5,  # This part is so annoying!! The spacing issue was because of this!!!
                     share_all=True,
                     cbar_mode=None)  

    plt.suptitle("Generated Images Across Checkpoints", fontsize=16, y=0.95)

    for row in range(num_prompts):
        for col in range(num_checkpoints):
            index = row * num_checkpoints + col
            ax = grid[index]
            ax.imshow(results[checkpoints[col]][row])
            ax.axis('off')

            # X-axis has the checkpoints (1000 to 15000 is too many, we should use a step size of 3000)
            if row == 0:
                ax.set_title(f"Step {checkpoints[col]}", fontsize=10)

            # Y-axis has the prompts we gave
            if col == 0:
                ax.set_ylabel(prompts[row], labelpad=80)

    plt.tight_layout(rect=[0, 0, 1, 0.9])  
    plt.subplots_adjust(top=0.2)  # This part makes the prompts not fall off the page!! (But the prompts might still be too long....)
    plt.savefig("wframe.png")
    plt.show()

def generate_gaussian_noise_image(width, height, mean=0, std_dev=1):
    """
    Generate a PIL image with Gaussian noise. (Just to check if our Visualize function is working)
    """
    noise = np.random.normal(mean, std_dev, (height, width, 3))
    noise = np.clip(noise, 0, 255)
    noise = noise.astype(np.uint8)

    noise_image = Image.fromarray(noise)
    return noise_image

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, help='Path where Trained RKS-Diffusion model checkpoints are stored')
    parser.add_argument("--step_size", type=int, help='Step size (1000, 2000) for the visualization generation')
    args = parser.parse_args()
    
    prompts = [
        " A majestic Bengal tiger with vibrant orange fur, stalking through a lush tropical rainforest dappled with sunlight", # EN
        "Un majestueux tigre du Bengale à la fourrure orange vif, traquant une forêt tropicale luxuriante parsemée de soleil", # DE
        "Ein majestätischer Bengaltiger mit leuchtend orangefarbenem Fell, der durch einen üppigen tropischen Regenwald stapft, der von Sonnenlicht durchflutet wird", # FR
    ]
    
    ckpt_path = args.checkpoint_path
    step_size = args.step_size
    
    aadit_rks_model_id = "AaditD/rks-lora-diffusion" # The Hugginface ID for our model!!
    
    info = model_info(aadit_rks_model_id)
    model_base = info.cardData["base_model"]
    print(model_base)   # CompVis/stable-diffusion-v1-4
    
    generator = torch.Generator("cuda").manual_seed(1)      

    pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    
    checkpoints = list(range(0, 15000, step_size))
    # random_image = generate_gaussian_noise_image(512, 512, mean=128, std_dev=20) # This was for debugging the visualize function
    checkpoint_image_mapper = {t:[] for t in checkpoints} # Key is Checkpoint, Value is the images Generated using this checkpoint (all the prompts)
    
    for timestep in checkpoints:
            
        print("Pipeline loaded!")
        if timestep != 0: 
            pipe.load_lora_weights(f"{ckpt_path}{timestep}")
            print("LoRA Weights Loaded")
        if timestep == 0: # The initial image just uses the normal Stable Diffusion model without our LoRA weights!
            print("Using Default Weights!")
        
        current_images = pipe(prompts, height=768, width=768, generator=generator).images
        checkpoint_image_mapper[timestep] = current_images
        print("Images Generated")
        
        os.makedirs(os.path.dirname(f"./poster/{timestep}/"), exist_ok=True)
        for i, im in enumerate(current_images, 0):
            im.save(f"./poster/{timestep}/a{i}.png")
            
        if timestep != 0: # Unload the LoRA weights after each checkpoint
            pipe.unload_lora_weights()
            print("LoRA Weights Unloaded")
        
if __name__ == "__main__":
    main()

    