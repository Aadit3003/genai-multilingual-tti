import torch
import time
import pandas as pd
from tqdm.auto import tqdm, trange
from huggingface_hub import model_info
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import wandb
from optimum.quanto import freeze, qfloat8, quantize

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch_type = torch.bfloat16
assert torch.cuda.is_available(), "Torch Cuda is NOT available!"

# Generation Hyperparameters!
height, width = 512, 512 # Image dimensions
num_inference_steps = 50  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance

def main():
    wandb_api = "df1248450b282ba9bdaf39161311b2d5c72ccad0"
    wandb.login(key=wandb_api)
    # Define device
    
    
    
    data_path = "../data/final_test.csv"
    ckpt_path = "/home/ubuntu/fine_tuned_rks_lora_new/"
    aadit_path = "AaditD/rks-lora-diffusion"
    
    info = model_info(aadit_path)
    model_base = info.cardData["base_model"]
    print(model_base) 
    
    pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    print("Pipeline loaded!")
    pipe.load_lora_weights(f"{ckpt_path}")
    print("LoRA Weights Loaded")
    
    # print()
    # print("*"*80)
    # print(pipe)
    # print("*"*80)
    # print()
    
    # Experimental and Highly Optional Quantization!!
    # quantize(pipe.text_encoder, weights=qfloat8)
    # freeze(pipe.text_encoder)
    
    
    
    wandb.init(
            name = f"finetuned_lora_rks_inference",
            reinit = True,
            project = "Gen-AI-Multilingual-TTI",
    )
    
    df = pd.read_csv(data_path)
    
    batch_size = 8
    prompts = df["caption_alt_text_description"].to_list()
    
    num_batches = len(prompts)//batch_size if len(prompts)%batch_size == 0 else len(prompts)//batch_size + 1
    print(f"NUMBER OF BATCHES: {num_batches}")

    start_time = time.time()
    save_path_prefix = f'/home/ubuntu/fine_tuned_lora_inferenced_images/'
    
    df[f"lora_save_paths"] = [f'{save_path_prefix}a{i}.png' for i in range(len(prompts))]
    
    data_save_path = data_path.split(".csv")[0]+"_with_lora_paths.csv"
    print(f"Saving DF with paths to {data_save_path}")
    df.to_csv(data_save_path)
    
    current_step = 0
    for i in trange(0, len(prompts), batch_size, desc=f"Batch Processing with LoRA Finetuned model"):
        batch_prompts = prompts[i : i+batch_size]
        
        images = pipe(batch_prompts, height=512, width=512, num_inference_steps=50).images
        
        for j, image in enumerate(images, 0):
            image.save(f'{save_path_prefix}a{i + j}.png')
    
        

        wandb.log({"batches_completed": i//batch_size + 1}, step=current_step)
        current_step += 1
    
    end_time = time.time()
    print(f"TOTAL TIME: {end_time - start_time} seconds!")
        
    
   
    print("ALL DONE!")
    wandb.finish()

if __name__ == "__main__":
    main()