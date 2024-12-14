from urllib.request import urlopen
from urllib.error import *
import pandas as pd
import numpy as np
import torch
import glob
import requests
import io
import os
import json
import sys
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import NllbTokenizer
from urllib.request import urlopen, Request
from PIL import Image
from transformers import AltCLIPModel, AltCLIPProcessor
from tqdm import tqdm
from utils.clip import get_clip_scores
from utils.translator import add_translation_column
from datasets import load_dataset
from huggingface_hub import login

device = "cuda"
torch_dtype = torch.float16
cache_path = "/ocean/projects/cis240008p/adeshpa2/genai/"

def get_lang_lengths(df):
    " df length per language"
    print("TOTAL: ", len(df))
    print("EN: ", len(df[df["language"] == "en"]))
    print("DE: ", len(df[df["language"] == "de"]))
    print("FR: ", len(df[df["language"] == "fr"]))

def filter_df(df, langs=['en', 'de', 'fr'], min_length=8, min_size=512, equalize=True):
    " First layer of Filtering by caption lengths, image sizes"
    #en/de/fr
    df = df[df['language'].isin(langs)]
    print("Initial counts: ", df.groupby('language').size(), sep='\n', end='\n\n')

    #nan captions
    df = df[df['caption_alt_text_description'].notna()]
    print("Remove NaN: ", df.groupby('language').size(), sep='\n', end='\n\n')

    #caption lengths
    df = df[df['caption_length'] >= min_length]
    print(f"Remove captions < {min_length} words: ", df.groupby('language').size(), sep='\n', end='\n\n')

    #height/width
    df = df[df['original_height'] >= min_size]
    df = df[df['original_width'] >= min_size]
    print(f"Remove image dims < {min_size}: ", df.groupby('language').size(), sep='\n', end='\n\n')

    #equal examples per lang
    if equalize:
        g = df.groupby('language')
        data_per_lang = min(g.size())
        print(f"{data_per_lang} examples per language; {data_per_lang * len(langs)} total")
        df : pd.DataFrame = g.apply(lambda x: x.iloc[:data_per_lang], include_groups=False)
        df = df.reset_index(level=0)

    return df

def is_url_valid(url):
    """ Check if a URL is valid! (Some aren't in WIT) """
    try:
      html = urlopen(url)

    except HTTPError as e:
      return False

    except URLError as e:
      return False

    else:
      return True

def filter_valid_urls(df):
    " Second layer of Filtering by Valid URLs"
    df['is_valid'] = df['image_url'].apply(is_url_valid)
    get_lang_lengths(df)
    print()

    df = df[df['is_valid']].drop(columns=['is_valid'])
    get_lang_lengths(df)
    return df

def filter_clip_scores(df, model, processor):
    urls = df["image_url"]
    captions = df["caption_alt_text_description"]
    df["CLIP_Sim"] = [get_clip_scores(u, c, model, processor) for u, c in tqdm(zip(urls, captions), total=len(urls))]
    df = df[df["CLIP_Sim"] >= 20] # CLIP Score Filtering
    return df

def download_image(url, save_path):
    """ Download images, because we need the image files too! While uploading to HF! """
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {save_path}")
    except Exception as e:
        print(f"Failed to download {url}. Error: {e}")

def process_dataset(df, output_dir):
    """ Make the df into a format that Huggingface ImageFolder likes! """
    
    os.makedirs(output_dir, exist_ok=True)

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns

    for _, row in df.iterrows():
        image_url = row["image_url"]
        file_name = row["file_name"]
        save_path = os.path.join(output_dir, file_name)
        download_image(image_url, save_path)

    # This is needed too!
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, "w") as f:
        for _, row in df.iterrows():
            json.dump(row.to_dict(), f)
            f.write("\n")
    print(f"Metadata saved to {metadata_path}")

def push_dataset(split, folder_path, name):
    """ Push dataset to HF! """
    dataset = load_dataset("imagefolder", data_dir=folder_path, split=split)

    dataset.push_to_hub(name)
    
    return dataset

def combine_tsvs(folder_path, file_pattern, output_file):
    """ Combine the TSVs in one downloaded folder into a single CSV """
    
    file_list = glob.glob(os.path.join(folder_path, file_pattern))
    print(f"Found {len(file_list)} files to combine.")
    combined_df = pd.concat([pd.read_csv(file, sep='\t') for file in file_list], ignore_index=True)


    language_counts = combined_df['language'].value_counts()
    min_count = language_counts.min()

    print(f"Languages found: {language_counts.to_dict()}")
    
    print(f"Sampling {min_count} rows per language.") # Sample equal rows per language!

    sampled_dfs = []
    for language in language_counts.index:
        sampled_dfs.append(combined_df[combined_df['language'] == language].sample(n=min_count, random_state=42))

    sampled_df = pd.concat(sampled_dfs, ignore_index=True)

    sampled_df.to_csv(output_file, index=False)
    print(f"Combined and sampled data saved to {output_file}.")
    
    return sampled_df

def main():
    a = 0
    # First download the train files from 0-9: https://github.com/google-research-datasets/wit/blob/main/DATA.md
    
    # Step 1: Combine the TSVs into a single CSV file: ./final_dataset.csv
    
    folder_path = "/Users/aaditdeshpande/Downloads/"
    output_file = "./final_dataset.csv"
    file_pattern = "/Users/aaditdeshpande/Downloads/wit_v1.train.all-0000*-of-00010.tsv"
    
    df = combine_tsvs(folder_path, file_pattern, output_file)
    
    # Step 2: Filter the Dataset
    clip_model = AltCLIPModel.from_pretrained("BAAI/AltCLIP",
        device_map=device,
        torch_dtype=torch_dtype,
        cache_dir=cache_path)
    clip_processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP", 
                                                cache_dir=cache_path)

    print("CUDA: ", torch.cuda.is_available())

    filename = sys.argv[1]

      # Step 2a: Filter by Caption lengths, image size
    df = filter_df(df)
      # Step 2b: Filter by Valid URLs
    df = filter_valid_urls(df)
      # Step 2c: Calculate the CLIP scores, and filter by them!
    df = filter_clip_scores(df, clip_model, clip_processor)
    
    # Step 3: Add NLLB-200 Translations for the German and French Captions
    languages = ["eng_Latn", "fra_Latn", "deu_Latn"]
    
    translation_model_name = "facebook/nllb-200-distilled-600M"
    trans_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name, cache_dir=cache_path).cuda()
    trans_tokenizer = NllbTokenizer.from_pretrained(translation_model_name, cache_dir=cache_path)

    subset_dfs = []
    for lang in languages:
    
        subset_dfs.append(add_translation_column(trans_model, trans_tokenizer, df, lang))
        print(f"{lang} DONE!!")
    
    df_translated = pd.concat(subset_dfs)
    
    # Step 4: Split the df into a train and test set!

    df_train, df_test = train_test_split(df_translated, test_size=0.2) # 6k / 1.5k
    df_test.to("final_test.csv")
    df_train.to("final_train.csv")

    test_output_dir = "../../folder_small/test"
    train_output_dir = "../../folder_small/train"
    
    process_dataset(test_output_dir)
    process_dataset(train_output_dir)
    
    # Step 5: Upload to Huggingface!
    
    login("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")

    train_dataset = push_dataset(split="train", folder_path="../../folder/", name="multilingual_rks") 
    test_dataset = push_dataset(split="test", folder_path="../../folder/", name="multilingual_rks")
    print()
    print("DATASET SAMPLE: ")
    print(train_dataset[0])
    print(test_dataset[0])

if __name__ == "__main__":
    main()  
    print("DONE!")
