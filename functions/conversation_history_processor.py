import json
import os
import logging
import yaml 

# Set up logging configuration
# logging.basicConfig(level=logging.INFO)

class ConversationHistoryProcessor:
    """
    A class to manage conversation history for a torque-controlled robot interface.
    """

    def __init__(self, system_role_content, conversation_history_file):
        """
        Initializes the ConversationHistoryProcessor with a specific system role content 
        and a unique conversation history file.
        """
        self.system_role_content = system_role_content
        self.conversation_history_file = conversation_history_file
        self.ensure_history_file()

    def ensure_history_file(self):
        """
        Ensures that the conversation history file exists.
        """
        if not os.path.exists(self.conversation_history_file):
            os.makedirs(os.path.dirname(self.conversation_history_file), exist_ok=True)
            with open(self.conversation_history_file, 'w') as f:
                json.dump([], f)

    def get_recent_conversation_history(self,num_messages=10):
        """
        Retrieves the recent conversation history along with the system role.
        """
        messages = []

        # Append system role to messages
        system_role = {
            "role": "system",
            "content": [{"type": "text", "text": self.system_role_content}]
        }
        messages.append(system_role)

        # Load conversation history
        try:
            with open(self.conversation_history_file, 'r') as f:
                data = json.load(f) or []
        except json.JSONDecodeError as e:
            logging.error(f"Error reading conversation history: {e}")
            data = []

        # Append last 10 items of data to messages
        messages.extend(data[-num_messages:])

        return messages

    def update_conversation_history(self, transcription, response, image_url=None, include_images=True):
        """
        Updates the conversation history with a new transcription and response.
        If image_url is provided, it includes the image in the message based on the include_images flag.

        Args:
            transcription (str): The transcription of the user's message.
            response (str): The system's response to the user.
            image_url (str, optional): The URL of the image to include, if any. Defaults to None.
            include_images (bool, optional): Flag to determine whether to include the image in the history. Defaults to True.
        """
        # Get recent conversation history excluding the system role
        recent_conversation_history = self.get_recent_conversation_history()[1:]

        # Create new message entry
        content = [{"type": "text", "text": transcription}]
        if image_url and include_images:
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        transcription_entry = {
            "role": "user",
            "content": content
        }
        response_entry = {
            "role": "system",
            "content": [{"type": "text", "text": response}]
        }

        # Append new messages to the conversation history
        recent_conversation_history.extend([transcription_entry, response_entry])

        # Save updated conversation history
        with open(self.conversation_history_file, 'w') as f:
            json.dump(recent_conversation_history, f)


    def reset_conversation_history(self):
        """
        Resets the conversation history by clearing the conversation history file.
        """
        with open(self.conversation_history_file, 'w') as f:
            json.dump([], f)
        logging.info(f"Conversation history in {self.conversation_history_file} has been reset.")
        return f"Conversation history in {self.conversation_history_file} has been reset."


if __name__ == "__main__":
    # The YAML file is at: experiment_data/system_roles/roles.yaml
    roles_yaml_path = os.path.join("experiment_data", "system_roles", "roles.yaml")
    with open(roles_yaml_path, 'r', encoding='utf-8') as f:
        roles_config = yaml.safe_load(f) # roles_config is now a dict with role0, role1, ..., role5

    # Example user transcription and system response (for testing)
    transcription_example = "Adjust the stiffness matrix for the new task."
    response_example = "Here is the adjusted stiffness matrix for your scenario."

    # Iterate over each role, create a conversation manager, and test updating/retrieving the history
    for role_key, role_info in roles_config.items():
        # Retrieve the conversation history file path and the system role content
        conversation_history_file = role_info["conversation_history_file"]
        system_role_content = role_info["system_role_content"]

        # Create an instance of the ConversationHistoryProcessor
        conv_manager = ConversationHistoryProcessor(
            system_role_content=system_role_content,
            conversation_history_file=conversation_history_file
        )

        print(f"\n=== Testing {role_key} ===")

        # Update the conversation history
        conv_manager.update_conversation_history(
            transcription=transcription_example,
            response=response_example
        )

        # Retrieve and print recent conversation history
        recent_history = conv_manager.get_recent_conversation_history()
        print("Recent conversation history:")
        print(json.dumps(recent_history, indent=2))

        # Reset the conversation history
        conv_manager.reset_conversation_history()
        print(f"Reset conversation history for {role_key}.\n")