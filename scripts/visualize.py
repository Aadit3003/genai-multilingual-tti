from manual_inference import generate_image_from_prompt
import torch
import os
from huggingface_hub import model_info
import textwrap
import time
import numpy as np
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

def visualize_results_with_imagegrid(results, prompts, checkpoints):
    """
    Visualizes the generated images in a grid layout using ImageGrid.

    Args:
        results (dict): A dictionary where keys are checkpoints (e.g., 1000, 3000, etc.)
                        and values are lists of 8 PIL images generated for prompts.
        prompts (list): List of 8 prompts used for generation.
        checkpoints (list): List of checkpoint numbers in ascending order.
    """
    num_prompts = len(prompts)
    num_checkpoints = len(checkpoints)

    # Create the figure and the grid
    fig = plt.figure(figsize=(num_checkpoints * 3, num_prompts * 3))
    grid = ImageGrid(fig, 111,  # 111: single subplot
                     nrows_ncols=(num_prompts, num_checkpoints),  # Grid dimensions
                     axes_pad=0.5,  # Space between images
                     share_all=True,  # Share x and y axes
                     cbar_mode=None)  # No colorbar for images

    # Add title for the entire figure
    plt.suptitle("Generated Images Across Checkpoints", fontsize=16, y=0.95)

    # Populate the grid with images
    for row in range(num_prompts):
        for col in range(num_checkpoints):
            index = row * num_checkpoints + col
            ax = grid[index]
            ax.imshow(results[checkpoints[col]][row])
            ax.axis('off')

            # Add column headers for checkpoints
            if row == 0:
                ax.set_title(f"Step {checkpoints[col]}", fontsize=10)

            # Add Y-axis labels manually for better visibility
            if col == 0:
                wrapped_prompt = "\n".join(textwrap.wrap(prompts[row], width=20))  # Adjust width for wrapping
                ax.annotate(wrapped_prompt, xy=(-0.1, 0.5),  # Adjust x-position for visibility
                            xycoords="axes fraction",
                            textcoords="offset points",
                            size="medium", ha="right", va="center", rotation=0)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leave less space for the title
    
    plt.subplots_adjust(top=0.2)  # Increase space on the left for Y-tick labels
    plt.savefig("wframe.png")
    plt.show()



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch_type = torch.bfloat16
assert torch.cuda.is_available(), "Torch Cuda is NOT available!"

def generate_gaussian_noise_image(width, height, mean=0, std_dev=1):
    """
    Generate a PIL image with Gaussian noise.

    Args:
        width (int): Width of the image.
        height (int): Height of the image.
        mean (float): Mean of the Gaussian noise.
        std_dev (float): Standard deviation of the Gaussian noise.

    Returns:
        PIL.Image: Image with Gaussian noise.
    """
    # Generate random Gaussian noise
    noise = np.random.normal(mean, std_dev, (height, width, 3))  # Shape (H, W, C) for RGB
    noise = np.clip(noise, 0, 255)  # Clip values to fit within valid range for image
    noise = noise.astype(np.uint8)  # Convert to uint8 type

    # Convert to PIL image
    noise_image = Image.fromarray(noise)
    return noise_image

# Example usage
random_image = generate_gaussian_noise_image(256, 256, mean=128, std_dev=20)

def main():
    # prompts = [
    #     "Painting of a portly gentleman in a powdered grey wig and richly embroidered clothes.",
    #     """Double-story building with hipped roof. 3 bays, symmetrical. Double verandah, plain roof, returned, with plenty of ornamental cast-iron detail. Hoods and keystones above windows. Front door 6 panels, leaded fanlight, stained glass. Sliding sash windows. Multicoloured front door. Built in 1839 for Dr James Christie, first full-time government doctor in the area. He was also town councillor, MP & director of Beaufort Bank. The building served as a police station 1898 till after the Anglo-Boer War. Type of site: House Current use: Backpackers.""",
    #     "An African-American man looks just right of the camera. His helmet and white jersey both have an orange \"S\" over \"F\" logo on them. The man's left arm is crossed over his body and his right is out of the picture. There is a black and orange glove on his left hand.",
    #     "Vue de la tour-porche d'une église ; esplanade au premier plan ; la mer en arrière plan.",
    #     "Photo de la tête d'un cheval au pelage bouclé dans un box paillé.", # CUNTY EXAMPLE!!!
    #     "La façade délabrée d'un aéroport, portant les lettres AEROPORT INTERNATIONAL DE KISANGANI.",
    #     "NASA-Bild von Vatoa Am unteren Bildrand ist der Nordteil von Vuata Vatoa zu erkennen",
    #     "Durch Winderosion geprägte Landschaft (Deflationsebene mit Steinpflaster) in der Ahaggar-Region in Algerien (links) und auf Chryse Planitia, Mars (rechts)."
    # ]
    
    prompts = [
        "five apples surrounding a dapper rubber duck",
        "Fünf Äpfel umgeben eine adrette Gummiente",
        "Cinq pommes entourant un canard en caoutchouc pimpant", 
    ]
    
    ckpt_path = "/home/ubuntu/fine_tuned_rks_lora_new/checkpoint-"
    aadit_path = "AaditD/rks-lora-diffusion"
    
    info = model_info(aadit_path)
    model_base = info.cardData["base_model"]
    print(model_base)   # CompVis/stable-diffusion-v1-4
    
    # from diffusers import DiffusionPipeline
    generator = torch.Generator("cuda").manual_seed(1)      

    pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    # pipe.load_lora_weights("AaditD/rks-lora-diffusion")
    
    checkpoints = list(range(0, 1, 2000))
    random_image = generate_gaussian_noise_image(512, 512, mean=128, std_dev=20)
    checkpoint_image_mapper = {t:[random_image]*len(prompts) for t in checkpoints}
    
    checkpoints = [0, 15000]
    for timestep in checkpoints:
            
        print("Pipeline loaded!")
        if timestep != 0:
            pipe.load_lora_weights(f"{ckpt_path}{timestep}")
            print("LoRA Weights Loaded")
        if timestep == 0:
            print("Using Default Weights!")
        
        current_images = pipe(prompts, height=768, width=768, generator=generator).images
        checkpoint_image_mapper[timestep] = current_images
        print("Images Generated")
        
        
        os.makedirs(os.path.dirname(f"./poster/{timestep}/"), exist_ok=True)
        for i, im in enumerate(current_images, 0):
            im.save(f"./poster/{timestep}/a{i}.png")
            
        if timestep != 0:
            pipe.unload_lora_weights()
            print("LoRA Weights Unloaded")
        
    

if __name__ == "__main__":
    main()
    # ckpt_path = "/home/ubuntu/fine_tuned_rks_lora_new/"
    # aadit_path = "AaditD/rks-lora-diffusion"
    
    # info = model_info(aadit_path)
    # model_base = info.cardData["base_model"]
    # print(model_base) 
    
    # pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe = pipe.to("cuda")
    # print("Pipeline loaded!")
    # pipe.load_lora_weights(f"{ckpt_path}/checkpoint-500")
    # print("LoRA Weights Loaded")
    
    
    # prompt = ["Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", "A cat sipping hot chocolate in a christmas Sweater"]
    # images = pipe(prompt).images
    # print(images)
    # print(type(images))
    # for i, im in enumerate(images, 0):
    #     im.save(f"./viz/a{i}.png")
    