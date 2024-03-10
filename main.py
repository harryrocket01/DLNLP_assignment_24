import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])


import pandas as pd
import json

pd.set_option("display.max_columns", 999)
pd.set_option("display.width", 999)


file_path = "Dataset/github-typo-corpus.v1.0.0.jsonl"

data_list = []

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        json_data = json.loads(line.strip())
        data_list.append(json_data)

df = pd.DataFrame(data_list)
df_normalized = pd.concat(
    [df.drop(columns=["edits"]), pd.json_normalize(df["edits"].explode())], axis=1
)

df_normalized = df_normalized[df_normalized["tgt.lang"].isin(["eng"])]

print(df_normalized)
print(df_normalized.columns.values.tolist())
print(df_normalized[["src.text", "tgt.text", "is_typo"]])
