from datasets import load_dataset

from huggingface_hub import login
login("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")

def push_dataset(split, folder_path, name):
    dataset = load_dataset("imagefolder", data_dir=folder_path, split=split)

    dataset.push_to_hub(name)
    
    return dataset

# train_dataset = push_dataset(split="train", folder_path="../../folder/", name="multilingual_rks") 
test_dataset = push_dataset(split="test", folder_path="../../folder/", name="multilingual_rks")
print()
print("DATASET SAMPLE: ")
print(test_dataset[0])

