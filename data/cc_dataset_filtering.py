""" The script for Filtering the Conceptual Captions Dataset, stored in /data/teacher_set.csv (Used for Stage-1 Training!) 

NOTE: This script is included for completeness, there is no need to run it. 
We uploaded our dataset to: https://github.com/Aadit3003/genai-multilingual-tti/blob/6be390afd58aef54fe1caa05d0343267a4d324a2/data/teacher_set.csv
"""

import pandas as pd
from collections import defaultdict
import numpy as np

def stats(list):
    """ Get the stats of any list """

    print(f"Min: {min(list)}")
    print(f"Max: {max(list)}")
    print(f"Mean: {round(np.mean(list), 2)}")
    print(f"Median: {round(np.median(list), 2)}")
    print(f"Std: {round(np.std(list), 2)}")
    print()

    return round(np.median(list), 2)

def main():
    df = pd.read_csv("val_file_marian_final.tsv", sep="\t")
    df.head()

    image_files = df["image_file"].to_list()
    captions = [(c, l) for c, l in zip(df["caption"].to_list(), df["lang_id"].to_list())]

    url_dict = defaultdict(list)

    for i, cl in zip(image_files, captions):
        c, l = cl
        url_dict[i].append((c, l)) # Use Image File as a Unique id for the Dict!

    print(len(df))
    print(len(url_dict))

    count = 0
    removal_count = 0
    original_keys = list(url_dict.keys())
    for url in original_keys:
        # print(url, url_dict[url])
        if len(url_dict[url]) != 4: # Sometimes Image Files are duplicated!
            url_dict.pop(url)
            removal_count += 1    
    print("Removed: ", removal_count)
    print(len(url_dict))

    final_dict = {
        "image_url": [],
        "source_caption": [],
        "target_caption": [],
        "lang_id": []
    }

    for url, captions in url_dict.items():
        final_dict["image_url"].append(url)
        final_dict["image_url"].append(url)
        assert len(captions) == 4
        for c, l in captions:
            if l == "de":
                final_dict["source_caption"].append(c)
                final_dict["lang_id"].append(l)
            elif l == "fr":
                final_dict["source_caption"].append(c)
                final_dict["lang_id"].append(l)
            elif l == "en":
                final_dict["target_caption"].append(c)
                final_dict["target_caption"].append(c)


    final_df = pd.DataFrame(final_dict)

    final_df.to_csv("teacher_set.csv", index=False)


    # Because the CLIP Text encoder doesn't accept anything above 77 tokens
    # Captions below 20 tokens aren't descriptive enough :(
    filtered_df = final_df[final_df["source_caption"].str.len() > 20]
    filtered_df = final_df[final_df["target_caption"].str.len() <= 77]

    stats([len(c) for c in filtered_df["source_caption"].to_list()])
    stats([len(c) for c in filtered_df["target_caption"].to_list()])
    
if __name__ == "__main__":
    main()
    print("CC 12 Filtering is Complete!")