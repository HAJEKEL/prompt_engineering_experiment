import os
import json
import yaml
import logging

from functions.conversation_history_processor import ConversationHistoryProcessor
from functions.speech_processor import SpeechProcessor
from functions.stiffness_matrix_processor import StiffnessMatrixProcessor

def main():
    logging.basicConfig(level=logging.INFO)

    # 1) Load roles from YAML
    roles_yaml_path = os.path.join("experiment_data", "system_roles", "roles.yaml")
    with open(roles_yaml_path, 'r', encoding='utf-8') as f:
        roles_config = yaml.safe_load(f)

    # 2) Prepare the stiffness matrix processor (for parsing GPT responses)
    stiffness_processor = StiffnessMatrixProcessor(matrices_dir="matrices")

    # 3) Load ground-truth messages (our "pre knowledge") from a JSON file
    ground_truth_file = os.path.join("experiment_data", "labels", "ground_truth_messages.json")
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth_messages = json.load(f)
        # This should be a list of dicts in the same conversation-message structure (role, content)

    # 4) Loop over each role and run a small test
    for role_key, role_info in roles_config.items():
        system_role_content = role_info["system_role_content"]
        conversation_history_file = role_info["conversation_history_file"]

        logging.info(f"\n--- Running experiment for {role_key} ---")

        # a) Create conversation manager, now including ground_truth_messages
        conv_manager = ConversationHistoryProcessor(
            system_role_content=system_role_content,
            conversation_history_file=conversation_history_file,
            pre_knowledge_messages=ground_truth_messages  # <-- NEW
        )
        conv_manager.reset_conversation_history()

        # b) (Optionally) create a speech processor using the same role & file
        #    so that it uses the same conversation manager under the hood.
        #    The SpeechProcessor itself also creates a ConversationHistoryProcessor,
        #    but for consistency, pass the same parameters:
        speech_processor = SpeechProcessor(
            system_role_content=system_role_content,
            conversation_history_file=conversation_history_file
        )

        # c) Create a test user prompt
        user_prompt = (
            "Please tell me what you see in the image. "
            "After that, please give me the stiffness matrix that aligns with the groove in the image."
        )

        # (If you have an image, you could set `test_image_url = "...some url..."`)
        test_image_url = "https://prompt-engineering-experiment.ngrok.io/images/1.jpg"

        # d) Get a GPT response (requires valid OpenAI credentials)
        gpt_response = speech_processor.get_gpt_response_vlm(
            transcript=user_prompt,
            image_url=test_image_url
        )

        if gpt_response:
            logging.info(f"Received GPT response for {role_key}:\n{gpt_response}")

            # e) Extract the stiffness matrix from the GPT response
            #    We pass 'role_key' because your method signature expects it
            matrix, matrix_path = stiffness_processor.extract_stiffness_matrix(gpt_response, role_key)
            if matrix:
                logging.info(f"Extracted stiffness matrix for {role_key}: {matrix}")
                logging.info(f"Matrix saved at: {matrix_path}")
            else:
                logging.warning(f"Could not extract a valid stiffness matrix for {role_key}.")
        else:
            logging.warning(f"No response obtained for {role_key}.")

        # Optionally reset conversation for next iteration if desired:
        # conv_manager.reset_conversation_history()

        logging.info(f"--- Finished experiment for {role_key} ---\n")


if __name__ == "__main__":
    main()
