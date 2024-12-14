""" Fine-tuning script for Training Stage 1: Teacher Learning (Here, we train the CLIP Text encoder on parallel multilingual data) """

from huggingface_hub import login
from huggingface_hub import create_repo, upload_folder
import torch
import gc
import PIL
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel,StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
import pandas as pd
from tqdm.auto import tqdm, trange
import numpy as np
import wandb
import os

CACHE_DIR = "/ocean/projects/cis240008p/adeshpa2/genai/"
DATASET_PATH = "/jet/home/adeshpa2/teacher_set.csv"
MODEL_SAVE_PATH = "/ocean/projects/cis240008p/adeshpa2/genai/fair_teacher_clip_models/"
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch_type = torch.bfloat16
assert torch.cuda.is_available(), "Torch Cuda is NOT available!"


diffusion_model_name = "stabilityai/stable-diffusion-2-1"
tokenizer = CLIPTokenizer.from_pretrained(diffusion_model_name, torch_dtype=torch_type, subfolder="tokenizer")

def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


## IMAGE GENERATION FUNCTIONS

height, width = 768, 768 # Image dimensions
num_inference_steps = 50  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance

def get_random_noise(batch_size: int, channel: int, height: int, width: int, generator: torch.Generator) -> torch.Tensor:
    '''Generate random noise of the specified shape.'''

    return torch.randn(size=(batch_size, channel, height, width), generator=generator, device=device, dtype=torch_type)

def generate_image_from_prompt(prompts:list, text_encoder, tokenizer, vae, unet, scheduler, batch_num=0):

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
        image.save(f'a{batch_num + j}.png')

    return images

class CustomTeacherDataset(Dataset):
  def __init__(self, teacher_set_file_path):
    self.parallel_caption_df = pd.read_csv(teacher_set_file_path)
  
  def __len__(self):
    return len(self.parallel_caption_df)

  def __getitem__(self, index):
    source_caption = self.parallel_caption_df.iloc[index, 1]
    target_caption = self.parallel_caption_df.iloc[index, 2]
    lang_id = self.parallel_caption_df.iloc[index, 3]
     
    return source_caption, target_caption

def preprocess_function(examples):
  source_captions = [s for s, t in examples]
  target_captions = [t for s, t in examples]
  
  tokenized_source_captions = tokenizer(source_captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
  tokenized_target_captions = tokenizer(target_captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
  return {
      "input_ids": tokenized_source_captions.input_ids,
      "attention_mask": tokenized_source_captions.attention_mask,
      "labels": tokenized_target_captions.input_ids  # Assuming you're using a seq2seq model
  }

def get_cls_embeddings(clip_model, text_input):
  """
  text_input can either be the tokenized_text_input_ids OR just plain text!
  """
  text_embeddings = clip_model(text_input.to(device))

  return text_embeddings.pooler_output

# ga_prompts = ["Dear Galinda you are just too good", "we don't mean to show a bias", "but Galinda you're a martyr"]
# a = get_cls_embeddings(student_text_encoder, ga_prompts)

def teacher_learning_loss_one_batch(teacher_text_encoder, student_text_encoder, source_texts, target_texts, criterion):
  teacher_cls = get_cls_embeddings(teacher_text_encoder, target_texts) # CLS tokens of English Captions in the batch (Teacher Model)
  student_cls = get_cls_embeddings(student_text_encoder, source_texts) # CLS tokens of DE/FR Captions in the batch (Student Model)

  loss_tensor = criterion(student_cls, teacher_cls)
  return loss_tensor


def train_loop(device, teacher_text_encoder, student_text_encoder, tokenized_teacher_dataloader, criterion, optimizer, lr_scheduler, progress_bar, epoch_number):
    
    student_text_encoder.train()
    current_epoch_train_loss = 0
    num_batches = len(tokenized_teacher_dataloader)
    dataset_size = num_batches * BATCH_SIZE 
    
    batch_number = 0
    for batch in tqdm(tokenized_teacher_dataloader, desc=f"Processing Epoch number {epoch_number}"):
        source_captions = batch["input_ids"].to(device)
        target_captions = batch["labels"].to(device)
            
            # P, L
        try:
            batch_loss = teacher_learning_loss_one_batch(teacher_text_encoder, student_text_encoder, source_captions, target_captions, criterion)
            current_epoch_train_loss += batch_loss.item()
                # B
            batch_loss.backward()

                # S
            optimizer.step()
            lr_scheduler.step()
                
                # Z
            optimizer.zero_grad()
            progress_bar.update()
            
            
                
        except RuntimeError as e: # OOM Error!
            optimizer.zero_grad(set_to_none=True)
            cleanup()
            print('OOM error, carry on!')
            continue
    
        

        if batch_number%1000 == 0:
            
            if batch_number > 0:
                checkpoint_path = f"{MODEL_SAVE_PATH}checkpoint-epoch-{epoch_number}-batch_number-{batch_number}"
                os.makedirs(checkpoint_path, exist_ok=True)
                student_text_encoder.save_pretrained(checkpoint_path)
                
                print(f"Saved Student Model after Batch {batch_number} in Epoch {epoch_number}!!")
        
        wandb.log({"batch_train_loss":batch_loss.item(), "num_train_examples":epoch_number*dataset_size + (batch_number+1) * BATCH_SIZE})
        
        batch_number += 1
    
    return current_epoch_train_loss / num_batches

def main():
    
    # LOGINS
    hub_token = "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"
    login(hub_token)
    wandb_api = "df1248450b282ba9bdaf39161311b2d5c72ccad0"
    wandb.login(key=wandb_api)
    wandb.init(
            name = f"Small Teacher",
            reinit = True,
            project = "Gen-AI-Multilingual-TTI",
    )
    
    ## MODELS AND DATASETS
    tokenizer = CLIPTokenizer.from_pretrained(diffusion_model_name, torch_dtype=torch_type, subfolder="tokenizer")
    # M
    teacher_text_encoder = CLIPTextModel.from_pretrained(diffusion_model_name, torch_dtype=torch_type, subfolder="text_encoder", cache_dir=CACHE_DIR).to(device)
    student_text_encoder = CLIPTextModel.from_pretrained(diffusion_model_name, torch_dtype=torch_type, subfolder="text_encoder", cache_dir=CACHE_DIR).to(device)

    teacher_dataset = CustomTeacherDataset(DATASET_PATH)
    tokenized_teacher_dataloader = DataLoader(dataset=teacher_dataset, batch_size=BATCH_SIZE, collate_fn=preprocess_function, shuffle=True)
    
    vae = AutoencoderKL.from_pretrained(diffusion_model_name, torch_dtype=torch_type, subfolder="vae", cache_dir=CACHE_DIR).to(device)
    unet = UNet2DConditionModel.from_pretrained(diffusion_model_name, torch_dtype=torch_type, subfolder="unet", cache_dir=CACHE_DIR).to(device)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(diffusion_model_name, torch_dtype=torch_type, subfolder="scheduler", cache_dir=CACHE_DIR)

    ## TRAINING PREREQUISITES
    # C
    criterion = nn.MSELoss()
    
    # O
    optimizer = AdamW(student_text_encoder.parameters(), lr=LEARNING_RATE) # O

    num_training_steps = NUM_EPOCHS * len(tokenized_teacher_dataloader)  # 2 * (25441//8 + 1)
    print("TRAINING STEPS: ", num_training_steps)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=num_training_steps)

    ## ACTUAL TRAINING LOOP
    progress_bar = tqdm(range(num_training_steps))
    
    validation_prompts = [
        "A majestic Bengal tiger with vibrant orange fur, stalking through a lush tropical rainforest dappled with sunlight", # EN
        "Un majestueux tigre du Bengale à la fourrure orange vif, traquant une forêt tropicale luxuriante parsemée de soleil", # FR
        "Golden eagle soaring above snowcapped mountain peaks", # EN
        "Steinadler schwebt über schneebedeckten Berggipfeln" # DE
    ]

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch}")
        
        epoch_train_loss = train_loop(device, teacher_text_encoder, student_text_encoder, tokenized_teacher_dataloader, criterion, optimizer, lr_scheduler, progress_bar, epoch)
        wandb.log({"epoch_train_loss":epoch_train_loss, "epoch":epoch})
        
        cleanup()
        
        ## INFERENCE WITH NEWLY TRAINED MODEL!!
        with torch.autocast(device_type="cuda"):
            images = generate_image_from_prompt(validation_prompts[0:4],
                                        student_text_encoder,
                                        tokenizer,
                                        vae,
                                        unet,
                                        scheduler)
        cleanup()

        wandb.log({"val_examples": [wandb.Image(image) for image in images], "val_prompts": validation_prompts, "epoch":epoch})


    ## PUSH MODEL TO HF!
    
    # Save final model
    student_text_encoder.save_pretrained(MODEL_SAVE_PATH)
    # student_text_encoder.push_to_hub("AaditD/rks-clip-text-encoder")
    hub_model_id = "AaditD/rks-clip-text-encoder"
    hub_token = "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"
    repo_id = create_repo(
                    repo_id=hub_model_id, exist_ok=True, token=hub_token
                ).repo_id

    upload_folder(
                repo_id=repo_id,
                folder_path=MODEL_SAVE_PATH,
                commit_message="End of training"
    )
    
    wandb.finish()

    
if __name__ == "__main__":
    main()