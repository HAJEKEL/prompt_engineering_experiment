import os
import json
import yaml
import logging

def load_roles():
    roles_yaml_path = os.path.join("experiment_data", "system_roles", "roles.yaml")
    with open(roles_yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_ground_truth_messages(category):
    ground_truth_file = os.path.join("experiment_data", "labels", f"ground_truth_messages_{category}.json")
    if os.path.exists(ground_truth_file):
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None  # Return None if no ground truth file exists for the category

def generate_prompt(system_role_key, image_resolution, ground_truth_category):
    roles_config = load_roles()
    
    system_role_key = f"role{system_role_key}"  # Convert numeric input to expected format
    
    if system_role_key not in roles_config:
        print(f"Error: System role '{system_role_key}' not found in roles.yaml.")
        return
    
    role_info = roles_config[system_role_key]
    system_role_content = role_info["system_role_content"]
    
    ground_truth_messages = load_ground_truth_messages(ground_truth_category) if ground_truth_category else None
    
    # Initialize history and ensure the system role is always the first message
    history = [
        {
            "role": "system",
            "content": system_role_content
        }
    ]
    
    if ground_truth_messages:
        history.extend(ground_truth_messages)
    
    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please describe the image and compute the stiffness matrix."},
            {"type": "image_url", "image_url": {"url": f"https://prompt-engineering-experiment.ngrok.io/images/sample.jpg", "detail": image_resolution}}
        ]
    }
    
    history.append(user_message)
    
    print("\nGenerated Messages Sent to OpenAI:\n")
    print(json.dumps(history, indent=2))

def main():
    logging.basicConfig(level=logging.INFO)
    
    print("Select System Role (1-5):")
    system_role_key = input().strip()
    
    print("Select Image Resolution (low/high):")
    image_resolution = input().strip().lower()
    if image_resolution not in ["low", "high"]:
        print("Invalid resolution. Choose 'low' or 'high'.")
        return
    
    print("Select Ground-Truth Category (none/home/lab):")
    ground_truth_category = input().strip().lower()
    if ground_truth_category not in ["none", "home", "lab"]:
        print("Invalid category. Choose 'none', 'home', or 'lab'.")
        return
    
    ground_truth_category = None if ground_truth_category == "none" else ground_truth_category
    
    generate_prompt(system_role_key, image_resolution, ground_truth_category)

if __name__ == "__main__":
    main()
