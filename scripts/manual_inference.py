import torch
import time
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
from torchvision.transforms import Normalize
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm, trange
import numpy as np
import wandb

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch_type = torch.bfloat16
assert torch.cuda.is_available(), "Torch Cuda is NOT available!"

# Generation Hyperparameters!
height, width = 512, 512 # Image dimensions
num_inference_steps = 50  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance

def get_random_noise(batch_size: int, channel: int, height: int, width: int, generator: torch.Generator) -> torch.Tensor:
    '''Generate random noise of the specified shape.'''
    
    return torch.randn(size=(batch_size, channel, height, width), generator=generator, device=device, dtype=torch_type)

def generate_image_from_prompt(prompts:list, text_encoder, tokenizer, vae, unet, scheduler, save_path_prefix, batch_num):
    
    generator = torch.Generator(device=device)
    # generator.manual_seed(1023)
    
    # 1. Tokenize and Encode the Prompt
    text_inputs = tokenizer(
        prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
        
    batch_size = len(prompts)
    # print(f"BATCH SIZE IS: {batch_size}")
    
    max_length = text_inputs.input_ids.shape[-1]

    uncond_inputs = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # 2. Generate the Starting Random Noise    
    latents = torch.randn(size=(batch_size, unet.config.in_channels, height//8, width//8), generator=generator, device=device, dtype=torch_type)
    
    # 3. Denoise the Image (Learned Reverse Process)
    latents = latents * scheduler.init_noise_sigma

    scheduler.set_timesteps(num_inference_steps)
    
    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        
        # The Noise Residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Classifier-free Guidance
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Step ahead in the Learned Reverse Process
        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]
        
    # 4. Decode the image

    latents = 1 / vae.config.scaling_factor * latents
    with torch.no_grad():
        images = vae.decode(latents).sample
        
    vae_scale_factor = 2**(len(vae.config.block_out_channels)-1)
    
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    
    images = image_processor.postprocess(images, output_type="pil")
    
    for j, image in enumerate(images, 0):
        image.save(f'{save_path_prefix}a{batch_num + j}.png')
    
    return images
    
    
def main():
    wandb_api = "df1248450b282ba9bdaf39161311b2d5c72ccad0"
    wandb.login(key=wandb_api)
    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load components
    data_path = "../data/final_dataset_translated.csv"
    diffusion_model_name = "stabilityai/stable-diffusion-2-1"
    diff_model_cute = diffusion_model_name.split("/")[-1]
    cache_dir = '/opt/dlami/nvme'
    
    
    wandb.init(
            name = f"{diff_model_cute}_inference",
            reinit = True,
            project = "Gen-AI-Multilingual-TTI",
    )
    
    text_encoder = CLIPTextModel.from_pretrained(diffusion_model_name, torch_dtype=torch_type, subfolder="text_encoder", cache_dir=cache_dir).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(diffusion_model_name, torch_dtype=torch_type, subfolder="tokenizer", cache_dir=cache_dir)
    vae = AutoencoderKL.from_pretrained(diffusion_model_name, torch_dtype=torch_type, subfolder="vae", cache_dir=cache_dir).to(device)
    unet = UNet2DConditionModel.from_pretrained(diffusion_model_name, torch_dtype=torch_type, subfolder="unet", cache_dir=cache_dir).to(device)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(diffusion_model_name, torch_dtype=torch_type, subfolder="scheduler", cache_dir=cache_dir)

    df = pd.read_csv(data_path)
    batch_size = 8
    prompts = df["translated_caption_alt_text"].to_list()
    
    num_batches = len(df)//batch_size if len(df)%batch_size == 0 else len(df)//batch_size + 1
    print(f"NUMBER OF BATCHES: {num_batches}")

    start_time = time.time()
    save_path_prefix = f'../generated_images/{diff_model_cute}/'
    df[f"{diff_model_cute}_save_paths"] = [f'{save_path_prefix}a{i}.png' for i in range(len(prompts))]
    df.to_csv(data_path.split(".csv")[0]+"_with_paths.csv")
    
    current_step = 0
    for i in trange(0, len(prompts), batch_size, desc=f"Batch Processing with {diff_model_cute}"):
        batch_prompts = prompts[i : i+batch_size]
        generate_image_from_prompt(prompts=batch_prompts,
                                    text_encoder=text_encoder,
                                    tokenizer=tokenizer,
                                    vae=vae,
                                    unet=unet,
                                    scheduler=scheduler,
                                    save_path_prefix=save_path_prefix,
                                    batch_num=i)
        wandb.log({"batches_completed": i//batch_size + 1}, step=current_step)
        current_step += 1
    
    end_time = time.time()
    print(f"TOTAL TIME: {end_time - start_time} seconds!")
        
   
    print("ALL DONE!")
    wandb.finish()

if __name__ == "__main__":
    main()