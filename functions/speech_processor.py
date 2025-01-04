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
    A class to handle speech-to-text (STT), response generation using OpenAI's API, and text-to-speech (TTS).
    """
    
    def __init__(self, system_role_content, conversation_history_file="messages/conversation_history_speech_processor.json",pre_knowledge_messages=None):
        """
        Optionally allow passing in the system role content and conversation file,
        so we can customize it if needed.
        """
        # Initialize the OpenAI client
        self.initialize_openai_client()

        # Initialize the conversation manager
        # If no custom role is provided, it falls back to a generic default.
        self.conversation_history_processor = ConversationHistoryProcessor(
            system_role_content=system_role_content,
            conversation_history_file=conversation_history_file, pre_knowledge_messages=pre_knowledge_messages
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
            # Get recent conversation history
            history = self.conversation_history_processor.get_recent_conversation_history()

            # Prepare user message
            content = [{"type": "text", "text": transcript}]
            if image_url:
                # Add a cache-busting parameter to ensure fresh requests
                image_url_with_cache = f"{image_url}?cache_bust={int(time.time())}"

                # Verify URL accessibility before proceeding
                response = requests.get(image_url_with_cache, timeout=20)
                if response.status_code != 200:
                    logging.error(f"Image URL {image_url_with_cache} is inaccessible with status code {response.status_code}")
                    return None

                content.append({"type": "image_url", "image_url": {"url": image_url_with_cache}})

            user_message = {
                "role": "user",
                "content": content
            }

            # Append the user message to the history
            history.append(user_message)
            logging.info("User message added to conversation history.")
            client = self.client
            # Call the OpenAI API
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=history,
                stream=True,
            )
            gpt_response = ""  # Initialize an empty string to accumulate the response

            
            # Loop over the chunks from the stream
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    gpt_response += content  # Accumulate the streamed content
            logging.info(f"GPT response received: {gpt_response}")

            # Update the conversation history with the assistant's response
            self.conversation_history_processor.update_conversation_history(transcript, gpt_response, image_url=image_url)

            return gpt_response

        except Exception as e:
            logging.error(f"Error in get_gpt_response_vlm: {e}")
            return None

if __name__ == "__main__":
    # Example usage/testing
    processor = SpeechProcessor()
    test_image_url = "https://www.xenos.nl/pub/cdn/582043/800/582043.jpg"
    test_transcript = "What is the stiffness matrix for a groove oriented along X? And what do you see in the image?"

    # Make a request to the GPT model (will fail if you don't have valid OpenAI creds)
    gpt_response = processor.get_gpt_response_vlm(test_transcript, image_url=test_image_url)
    if gpt_response:
        print("GPT Response:", gpt_response)