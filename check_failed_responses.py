import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define the directory containing the conversation logs
CONVERSATION_LOGS_DIR = "conversation_logs"

def check_conversation_logs():
    empty_count = 0
    non_empty_count = 0
    total_files = 0
    
    if not os.path.exists(CONVERSATION_LOGS_DIR):
        logging.error(f"Directory '{CONVERSATION_LOGS_DIR}' does not exist.")
        return
    
    # Iterate through all JSON files in the directory
    for filename in os.listdir(CONVERSATION_LOGS_DIR):
        if filename.endswith(".json"):
            total_files += 1
            file_path = os.path.join(CONVERSATION_LOGS_DIR, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Check if the JSON content is an empty list
                    if isinstance(data, list) and len(data) == 0:
                        empty_count += 1
                    else:
                        non_empty_count += 1
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON file: {filename}")
            except Exception as e:
                logging.warning(f"Error processing file {filename}: {e}")
    
    # Print results
    logging.info(f"Total JSON files checked: {total_files}")
    logging.info(f"Files with empty lists: {empty_count}")
    logging.info(f"Files with non-empty lists: {non_empty_count}")

if __name__ == "__main__":
    check_conversation_logs()