import pandas as pd
import json
import os

# Load the JSONL file into a pandas DataFrame
save_path = "../../folder/train/"

# def check_all_files_are_there(save_path):
#     files = os.listdir(save_path)


#     print(len(files))
#     images = [f.split(".png")[0] for f in files if f.split(".png")[0].isnumeric()]

#     print(len(images))

#     expected_files = [f'{i:04}' for i in range(1, 1501)]

#     print(len(expected_files))

#     missing_file = list(set(expected_files) - set(images))
#     print(f'The missing file is: {missing_file}')
    
df = pd.read_json(f"{save_path}metadata.jsonl", lines=True)

# Replace NaN with None (which is converted to null in JSON)
df = df.where(pd.notnull(df), None)

# Save the DataFrame back as a JSONL file
with open(f"{save_path}metadata_fixed.jsonl", "w") as f:
    for _, row in df.iterrows():
        json.dump(row.to_dict(), f)
        f.write("\n")
