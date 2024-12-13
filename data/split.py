import pandas as pd
from sklearn.model_selection import train_test_split


def sample_n(size="small", test_proportion=0.2):
    
    df = pd.read_csv(f"./{size}_dataset_translated.csv")
    print("Original Length: ", len(df))
    langs = list(set(df["language"]))
    train_subset = []
    test_subset = []    
    
    for lang in langs:
        lang_df = df[df["language"]==lang]
        df_train, df_test = train_test_split(lang_df, test_size=test_proportion, random_state=42)
        print(len(df_train), len(df_test))
        train_subset.append(df_train)
        test_subset.append(df_test)
    
    df_train_final = pd.concat(train_subset)
    df_train_final["file_name"] = [f'{i}'.zfill(4) + '.png' for i in range(1, len(df_train_final)+1)]
    df_test_final = pd.concat(test_subset)
    df_test_final["file_name"] = [f'{i}'.zfill(4) + '.png' for i in range(1, len(df_test_final)+1)]
    
    df_train_final.to_csv(f"./{size}_train.csv", index=False)
    df_test_final.to_csv(f"./{size}_test.csv", index=False)

print("DONE!")

sample_n("small", 0.2)
sample_n("final", 0.2)
print([f'{i}'.zfill(4) + '.png' for i in range(1, 4+1)])