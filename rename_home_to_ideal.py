import json

# Define the mapping for conversation_prior
prior_mapping = {
    "home": "ideal"
}

# Load the JSON file
input_file = "experiment_results.json"  # Replace with your actual file name
output_file = "experiment_results_renamed_home_ideal.json"  # New file with modified values

with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Update conversation_prior values
for entry in data:
    if entry["conversation_prior"] in prior_mapping:
        entry["conversation_prior"] = prior_mapping[entry["conversation_prior"]]

# Save the modified JSON back to a file
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=2)

print(f"Updated JSON saved to {output_file}")
