import torch
from tqdm import tqdm
import pandas as pd
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def main():
    print(f"TORCH CUDA: {torch.cuda.is_available()}")
    model_id = "stabilityai/stable-diffusion-2-1"
    cache_dir = '/home/ubuntu/.cache/huggingface'

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=cache_dir)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    df = pd.read_csv("./small_dataset.csv").head(5)
    # prompt = "a photo of an astronaut riding a horse on mars"
    prompts = df["caption_attribution_description"]
    for i, prompt in tqdm(enumerate(prompts, 0)):
        image = pipe(prompt).images[0]
        print(f"a{i} Prompt: {prompt}")
        image.save(f"pipe_trial_outputs/a{i}.png")
        
    print("ALL DONE!")

if __name__ == "__main__":
    main()