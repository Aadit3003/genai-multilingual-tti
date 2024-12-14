""" Script to add the CLIP Scores to the image-caption pairs in WIT. 
Used by wit_dataset_filtering.py to filter the dataset

NOTE: This script is included for completeness, there is no need to run it. 
We uploaded our dataset to: https://huggingface.co/datasets/AaditD/multilingual_rks
"""
import torch
import requests
import io
import sys
from urllib.request import urlopen, Request
import PIL
from PIL import Image
from transformers import AltCLIPModel, AltCLIPProcessor
from tqdm import tqdm
import pandas as pd

device = "cuda"
torch_dtype = torch.float16
cache_path = "/ocean/projects/cis240008p/adeshpa2/genai/"

def get_clip_scores(url, captions, model, processor):
    """ Generates CLIP scores for an Image URL and its caption """
    try:
        req = Request(url, headers={'User-Agent': 'My CLIP Score Script'}) # Not including User-Agent causes FileNotFound errors!
        
        image = Image.open(urlopen(req))
        inputs = processor(text=captions, images=image, return_tensors="pt", padding=True)
        inputs.to(device)

        with torch.no_grad():
            with torch.autocast(device):
                outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        return logits_per_image.item()

    except (requests.exceptions.RequestException, PIL.UnidentifiedImageError) as e:
        print(f"Error processing URL {url}: {e}")
        return 666.0  # Filter this value later! CLIP Scores can't be this high, so it's ok!
