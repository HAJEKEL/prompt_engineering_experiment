import json

def count_dictionaries(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, list):
                print(f"Number of dictionaries in the list: {len(data)}")
            else:
                print("The JSON file does not contain a list.")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading the JSON file: {e}")

if __name__ == "__main__":
    count_dictionaries("experiment_results.json")
