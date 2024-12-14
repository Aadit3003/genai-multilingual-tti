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
from PIL import Image
import os
import argparse
from tqdm import tqdm
import pandas as pd

BATCH_SIZE = 64

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


def scale_images(images, new_shape):
	images_list = []
	for image in images:
		new_image = resize(np.array(image), new_shape, 0)
		images_list.append(new_image)
	return np.array(images_list)



def preprocess_images_uint8(images, image_size=299):
    """
    Preprocess images to torch.uint8 format for feeding into metrics calculation.
    Ensures images are in RGB format and resized to the same shape.
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
    
    if len(processed) == 0:
        raise RuntimeError("No valid images after preprocessing.")
    shapes = [img.shape for img in processed]
    if len(set(shapes)) > 1:
        raise RuntimeError(f"Inconsistent image shapes: {shapes}")
    
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
    # dataset_file = '../data/final_dataset_translated_with_paths.csv'
    path = '/opt/dlami/nvme/rks-diffusion-inferenced-images'
    n, start = 500, 1000
    fake_images = load_images_from_folder(path, start, start + n)
    print("Loaded fake images")
    
    # dataset = load_dataset('AaditD/multilingual_rks', split='test')

    # test_dataset = dataset# print(test_dataset[0:1])

    # # print(len(test_dataset))

    # # lang_fake_images = fake_images_from_folder[]
    # # print("Loading Real Images")
    # # lang_real_images = [example['image'] for example in test_dataset if example['language'] == 'de'][:n]

    # lang_real_images = [example['image'] for example in test_dataset.select(range(start, start + n))]
    # print("Loaded real images")

    # print('Starting eval...')
    is_avg_fake, is_std_fake = calculate_inception_score(fake_images, n_split=10)

    # print(f"Inception Score (fake): avg={is_avg_fake}, std={is_std_fake}")

    # fid_score = calculate_fid(lang_real_images, fake_images)
    # print(f"FID score: {fid_score}")
