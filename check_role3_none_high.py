import json
import pandas as pd
import os
import sys

# Define the JSON file path
json_file = "experiment_results.json"

# Load the JSON data
if not os.path.exists(json_file):
    print(f"Error: File '{json_file}' not found.")
    sys.exit(1)

with open(json_file, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Rename conversation_prior to prior if necessary
if "conversation_prior" in df.columns:
    df.rename(columns={"conversation_prior": "prior"}, inplace=True)

# Drop rows that have no matrix or metrics
df = df.dropna(subset=["estimated_matrix", "metrics"]).reset_index(drop=True)

# Extract the 'correct' field from the metrics dictionary
def extract_correct(metrics):
    if isinstance(metrics, dict) and "correct" in metrics:
        return metrics["correct"]
    return False

df["correct"] = df["metrics"].apply(extract_correct)

df = df.dropna(subset=["correct"]).reset_index(drop=True)

df["correct"] = df["correct"].astype(bool)

# Filter for role3 | none | high
df_filtered = df[
    (df["role"] == "role3") & 
    (df["prior"] == "none") & 
    (df["resolution"] == "high")
]

# Save filtered data to a new JSON file
filtered_json_file = "filtered_role3_none_high.json"
df_filtered.to_json(filtered_json_file, orient="records", indent=4)

print(f"Filtered data saved to '{filtered_json_file}' with {len(df_filtered)} rows.")
