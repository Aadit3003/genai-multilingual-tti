""" Evaluation Script for the generated images (Calculates FID score and IS score using the real and generated images) 

Run this only after running the training scripts (teacher_learning.py and train_text_to_image_lora_rks.py) which stores
the checkpoints needed to visualize the results """
import numpy as np
from skimage.transform import resize
from PIL import Image
import os
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import os
import argparse
import pandas as pd

def load_images_from_folder(folder_path, n_start, n_end):
	"""
	Load all images from a specified folder.
	"""
	images = []
	files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
		
	for img_path in files[n_start:n_end]:
		img = Image.open(img_path).convert('RGB')
		images.append(img)
	return images

def preprocess_images_uint8(images, image_size=299):
    """
    If we don't do this, it causes TypeErrors with torchmetrics.fidelity!!
    """
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),  
        transforms.Lambda(lambda img: img.convert("RGB")),  
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: (x * 255).byte()) 
    ])
    processed = []

    for image in images:
        try:
            processed_img = preprocess(image).unsqueeze(0)  
            processed.append(processed_img)
        except Exception as e:
            print(f"Error processing image: {e}")
    
    return torch.cat(processed)

def calculate_inception_score(images, n_split=10, device='cuda'):
    """
    Torchmetrics version of Inception Score (unit8 to avoid type errors) : Use this with Fake Images ONLY!
    """
    inception = InceptionScore(splits=n_split, normalize=False).to(device)

    images = preprocess_images_uint8(images).to(device)

    inception.update(images)

    is_avg, is_std = inception.compute()
    print(f"Inception Score: avg={is_avg.item()}, std={is_std.item()}")
    return is_avg.item(), is_std.item()

def calculate_fid(real_images, fake_images, device='cuda'):
    """
    Torchmetrics version of FID Score (unit8 to avoid type errors) : Use this with Real and Fake Images BOTH!
    """
    fid = FrechetInceptionDistance(feature=768, normalize=False).to(device)

    if len(real_images) < 2:
        raise RuntimeError("At least two real images are required to compute FID")
    if len(fake_images) < 2:
        raise RuntimeError("At least two fake images are required to compute FID")

    real_images = preprocess_images_uint8(real_images).to(device)
    fake_images = preprocess_images_uint8(fake_images).to(device)

    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    fid_score = float(fid.compute())
    print(f"FID: {fid_score}")
    return fid_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_image_dir", type=str, help='The directory containing the generated images we want to evaluate!')
    args = parser.parse_args()
    
    generated_image_dir = args.generated_image_dir
    
    path = generated_image_dir
    n_images_per_lang, starting_point = 500, 1000 # Consider setting n_images_per_lang to 200, if 500 causes an OOM error! (This crashes the EC2 server a lot!)
    

    starting_point = 0
    for lang in ['German (DE)', 'English (EN)', 'French (FR)']:
        print(f"Currently processing {lang}")
        
        fake_images = load_images_from_folder(path, starting_point, starting_point + n_images_per_lang)
        print("Loaded fake images")
        
        test_dataset = load_dataset('AaditD/multilingual_rks', split='test')
        lang_real_images = [example['image'] for example in test_dataset.select(range(starting_point, starting_point + n_images_per_lang))]
        print("Loaded real images")
        print()
        print('Starting evaluation')
        
        is_avg_fake, is_std_fake = calculate_inception_score(fake_images, n_split=10)
        fid_score = calculate_fid(lang_real_images, fake_images)
        
        starting_point += 500
        
        print("*"*80)