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
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from transformers import NllbTokenizer
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
# import sacrebleu
# from sacrebleu.metrics import BLEU, CHRF
import pandas as pd
# from comet import download_model, load_from_checkpoint

def translate(
    model, 
    tokenizer,
    text,
    src_lang='fra_Latn',
    tgt_lang='eng_Latn',
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
    """Translate texts in batches of similar length"""
    
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
    # print(lang_sentences)
    # print()
    
    if lang == "eng_Latn":
        translated_english_sentences = lang_sentences
    else:
        translated_english_sentences = batched_translate(model, tokenizer, texts=lang_sentences, src_lang=lang, tgt_lang="eng_Latn")
    
    df_subset["translated_caption_alt_text"] = translated_english_sentences
    
    return df_subset

def main():
    languages = ["eng_Latn", "fra_Latn", "deu_Latn"]

    df_full = pd.read_csv("./final_dataset.csv")

    model_name = "facebook/nllb-200-distilled-600M"
    cache_dir = '/opt/dlami/nvme'

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir).cuda()
    tokenizer = NllbTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    subset_dfs = []
    for lang in languages:
    
        subset_dfs.append(add_translation_column(model, tokenizer, df_full, lang))
        print(f"{lang} DONE!!")
    
    df_translated = pd.concat(subset_dfs)
    df_translated.to_csv(f"./final_dataset_translated.csv")
    print("ALL DONE!")
    

if __name__ == '__main__':
    main()