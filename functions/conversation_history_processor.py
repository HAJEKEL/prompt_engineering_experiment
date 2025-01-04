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

    def __init__(self, system_role_content, conversation_history_file,pre_knowledge_messages=None):
        """
        Initializes the ConversationHistoryProcessor with a specific system role content 
        and a unique conversation history file.
        """
        self.system_role_content = system_role_content
        self.conversation_history_file = conversation_history_file
        self.pre_knowledge_messages = pre_knowledge_messages or []  # <-- NEW: store the extra messages
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

        # 2) Append optional pre-knowledge messages  # <-- NEW BLOCK
        for msg in self.pre_knowledge_messages:
            messages.append(msg)

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
        skip_count = 1 + len(self.pre_knowledge_messages)  # <-- NEW

        # Get recent conversation history excluding the system role
        recent_conversation_history = self.get_recent_conversation_history()[skip_count:]

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
    # Demonstration of usage
    ground_truth_msgs = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "GroundTruth: For image 1, stiffness is 100 in x,y,z. For image 2, 250 in y, etc."
                }
            ]
        }
    ]

    conv_manager = ConversationHistoryProcessor(
        system_role_content="You are an expert on stiffness matrices.",
        conversation_history_file="messages/test_history.json",
        pre_knowledge_messages=ground_truth_msgs  # <-- NEW: pass the ground-truth or any extra data here
    )

    conv_manager.reset_conversation_history()
    conv_manager.update_conversation_history(
        transcription="What is the stiffness for image 1?",
        response="Ground truth indicates 100 in all directions."
    )

    history = conv_manager.get_recent_conversation_history()
    print(json.dumps(history, indent=2))