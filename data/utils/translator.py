""" Script to add the English Translations to the Original FR/DR captions in WIT
Used by wit_dataset_filtering.py to translate the captions

NOTE: This script is included for completeness, there is no need to run it. 
We uploaded our dataset to: https://huggingface.co/datasets/AaditD/multilingual_rks
"""
import re
import sys
import gc
import random
import numpy as np
import unicodedata
from tqdm.auto import tqdm, trange
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import NllbTokenizer
import pandas as pd
# from comet import download_model, load_from_checkpoint

def translate(model, tokenizer, text, src_lang='fra_Latn', tgt_lang='eng_Latn',
    a=32,
    b=3,
    max_input_length=1024,
    num_beams=4,
    **kwargs
):
    """ Turn a list of texts into a list of translations"""
    
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)
    
    model.eval()
    
    result = model.generate(**inputs.to(model.device),
                            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                            max_new_tokens=int(a+b*inputs.input_ids.shape[1]),
                            num_beams=num_beams, **kwargs
                            
                            )
    
    return tokenizer.batch_decode(result, skip_special_tokens=True)

def batched_translate(model, tokenizer, texts, batch_size=16, **kwargs):
    """ Translate texts in batches """
    
    # idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    
    for i in trange(0, len(texts), batch_size):
        results.extend(translate(model, tokenizer, texts[i:i+batch_size], **kwargs))
    
    return results

def add_translation_column(model, tokenizer, df, lang):
    df_subset = df[df["language"]==lang[0:2]]
    lang_sentences = df_subset["caption_alt_text_description"].to_list()
    print()
    print("CURRENT LANG: ", lang)
    
    if lang == "eng_Latn":
        translated_english_sentences = lang_sentences
    else:
        translated_english_sentences = batched_translate(model, tokenizer, texts=lang_sentences, src_lang=lang, tgt_lang="eng_Latn")
    
    df_subset["translated_caption_alt_text"] = translated_english_sentences
    
    return df_subset
