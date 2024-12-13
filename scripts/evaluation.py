#IS imports
import numpy as np
#import tensorflow as tf
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp
from PIL import Image
import os

#FID imports
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
#from torchvision.models import inception_v3
from PIL import Image
import os
import argparse
from tqdm import tqdm
import pandas as pd

from datasets import load_dataset

BATCH_SIZE = 64

#Loading images
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


#IS score
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
        transforms.Resize((image_size, image_size)),  # Resize to target dimensions
        transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure 3 channels (RGB)
        transforms.ToTensor(),  # Convert to tensor (float32)
        transforms.Lambda(lambda x: (x * 255).byte())  # Scale to [0, 255] and convert to uint8
    ])
    processed = []

    for image in images:
        try:
            processed_img = preprocess(image).unsqueeze(0)  # Add batch dimension
            processed.append(processed_img)
        except Exception as e:
            print(f"Error processing image: {e}")
    
    # Ensure all tensors have the same shape
    if len(processed) == 0:
        raise RuntimeError("No valid images after preprocessing.")
    shapes = [img.shape for img in processed]
    if len(set(shapes)) > 1:
        raise RuntimeError(f"Inconsistent image shapes: {shapes}")
    
    return torch.cat(processed)


def calculate_inception_score(images, n_split=10, device='cuda'):
    """
    Compute Inception Score using torchmetrics with corrected preprocessing for uint8 input.

    Args:
        images (list): List of PIL images.
        n_split (int): Number of splits for calculating Inception Score.
        device (str): Device to run the calculation ('cuda' or 'cpu').

    Returns:
        tuple: (mean IS, std IS)
    """
    # Initialize InceptionScore metric
    inception = InceptionScore(splits=n_split, normalize=False).to(device)

    # Preprocess images for uint8 format
    images = preprocess_images_uint8(images).to(device)

    # Update metric with images
    inception.update(images)

    # Calculate average and std deviation of scores
    is_avg, is_std = inception.compute()
    print(f"Inception Score: avg={is_avg.item()}, std={is_std.item()}")
    return is_avg.item(), is_std.item()

def calculate_fid(real_images, fake_images, device='cuda'):
    """
    Compute FID using torchmetrics with corrected preprocessing for uint8 input.
    """
    # Initialize the FrechetInceptionDistance metric
    fid = FrechetInceptionDistance(feature=768, normalize=False).to(device)

    # Ensure that there are enough images to calculate FID
    if len(real_images) < 2:
        raise RuntimeError("At least two real images are required to compute FID")
    if len(fake_images) < 2:
        raise RuntimeError("At least two fake images are required to compute FID")

    # Preprocess images for uint8 format
    real_images = preprocess_images_uint8(real_images).to(device)
    fake_images = preprocess_images_uint8(fake_images).to(device)

    # Update FID with real and fake images
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    # Compute and print the FID
    fid_score = float(fid.compute())
    print(f"FID: {fid_score}")
    return fid_score

def score_images(dataset_file, real_folder, generated_folder, n = 50):
    df = pd.read_csv(dataset_file)
    df_en = df[df['language'] == 'en']
    df_fr = df[df['language'] == 'fr']
    df_de = df[df['language'] == 'de']
    
    real_images = load_images_from_folder(real_folder, n)
    fake_images = load_images_from_folder(generated_folder, n)


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
