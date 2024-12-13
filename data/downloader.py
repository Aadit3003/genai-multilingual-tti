import os
import requests
import pandas as pd
import json

# Function to download an image with a custom User-Agent header
def download_image(url, save_path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {save_path}")
    except Exception as e:
        print(f"Failed to download {url}. Error: {e}")

# Function to process the dataset and download images
def process_dataset(csv_path, output_dir):
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV and drop unnamed columns
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns

    # Iterate through rows and download images
    for _, row in df.iterrows():
        image_url = row["image_url"]
        file_name = row["file_name"]
        save_path = os.path.join(output_dir, file_name)
        download_image(image_url, save_path)

    # Save metadata as JSONL
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, "w") as f:
        for _, row in df.iterrows():
            json.dump(row.to_dict(), f)
            f.write("\n")
    print(f"Metadata saved to {metadata_path}")

# Paths for test and train datasets
test_csv_path = "small_test.csv"
train_csv_path = "small_train.csv"

test_output_dir = "../../folder_small/test"
train_output_dir = "../../folder_small/train"

# Process both datasets
process_dataset(test_csv_path, test_output_dir)
process_dataset(train_csv_path, train_output_dir)

