import os
import sys
import json
import uuid
import time
import logging
import requests
import re
from pathlib import Path
import ffmpeg

# OpenAI API Imports
import openai
from decouple import config, RepositoryEnv

# Import the ConversationManager class
from functions.conversation_history_processor import ConversationHistoryProcessor

class SpeechProcessor:
    """
    A class to handle speech-to-text (STT), response generation using OpenAI's API, 
    and text-to-speech (TTS).
    """
    
    def __init__(
        self,
        system_role_content=None,
        conversation_history_file="messages/conversation_history_speech_processor.json",
        pre_knowledge_messages=None,
        image_resolution="high"
    ):
        """
        Optionally allow passing in the system role content, conversation file,
        and an initial set of pre-knowledge messages. 
        Also allows specifying image_resolution: "low" or "high".
        """
        # Initialize the OpenAI client
        self.initialize_openai_client()

        # Store the image resolution for later use
        self.image_resolution = image_resolution

        # Initialize the conversation manager
        self.conversation_history_processor = ConversationHistoryProcessor(
            system_role_content=system_role_content or "You are a generic interface for stiffness matrix adjustment.",
            conversation_history_file=conversation_history_file,
            pre_knowledge_messages=pre_knowledge_messages
        )

    @staticmethod
    def add_parent_to_sys_path():
        """
        Adds the parent directory to sys.path for module imports.
        """
        parent_dir = Path(__file__).resolve().parent
        if str(parent_dir) not in sys.path:
            sys.path.append(str(parent_dir))
            logging.info(f"Added {parent_dir} to sys.path")

    def initialize_openai_client(self):
        """
        Initializes the OpenAI API client.
        """
        organization = config("OPEN_AI_ORG")
        api_key = config("OPEN_AI_KEY")
        self.client = openai.OpenAI(api_key=api_key, organization=organization)
        logging.info("OpenAI client initialized successfully.")

    def get_gpt_response_vlm(self, transcript, image_url=None):
        """
        Generates a response using OpenAI's GPT model, optionally including an image.

        Parameters:
            transcript (str): The user's input text.
            image_url (str, optional): URL of the image to include in the prompt.

        Returns:
            str: The generated response from GPT.
        """
        try:
            # 1. Get recent conversation history
            history = self.conversation_history_processor.get_recent_conversation_history()

            # 2. Prepare user message
            content = [{"type": "text", "text": transcript}]
            if image_url:
                # Add a cache-busting parameter to ensure fresh requests
                image_url_with_cache = f"{image_url}?cache_bust={int(time.time())}"

                # Verify URL accessibility before proceeding
                resp = requests.get(image_url_with_cache, timeout=20)
                if resp.status_code != 200:
                    logging.error(
                        f"Image URL {image_url_with_cache} is inaccessible with status code {resp.status_code}"
                    )
                    return None

                # Include resolution detail in the message, e.g., "low" or "high"
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url_with_cache,
                        "detail": self.image_resolution  # "high" or "low"
                    }
                })

            user_message = {
                "role": "user",
                "content": content
            }

            # 3. Append the user message to the conversation history
            history.append(user_message)
            logging.info("User message added to conversation history.")

            # 4. Call the OpenAI API
            client = self.client
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=history,
                stream=True,
            )

            # 5. Accumulate the streamed response
            gpt_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    gpt_response += chunk.choices[0].delta.content

            logging.info(f"GPT response received: {gpt_response}")

            # 6. Update the conversation history with the assistant's response
            self.conversation_history_processor.update_conversation_history(
                transcription=transcript,
                response=gpt_response,
                image_url=image_url
            )

            return gpt_response

        except Exception as e:
            logging.error(f"Error in get_gpt_response_vlm: {e}")
            return None


if __name__ == "__main__":
    # Example usage/testing
    processor = SpeechProcessor(
        system_role_content="You are a matrix expert.",
        image_resolution="low"   # you can switch to "high"
    )
    test_image_url = "https://www.xenos.nl/pub/cdn/582043/800/582043.jpg"
    test_transcript = "What is the stiffness matrix for a groove oriented along X? And what do you see in the image?"

    # Make a request to the GPT model
    gpt_response = processor.get_gpt_response_vlm(test_transcript, image_url=test_image_url)
    if gpt_response:
        print("GPT Response:", gpt_response)
