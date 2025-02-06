#!/usr/bin/env python3
import os
import json
import yaml
import logging
import numpy as np
from pathlib import Path
import time

from functions.conversation_history_processor import ConversationHistoryProcessor
from functions.speech_processor import SpeechProcessor
from functions.stiffness_matrix_processor import StiffnessMatrixProcessor
from functions.stiffness_matrix_evaluator import StiffnessMatrixEvaluator

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

    # 1) Load roles from YAML
    roles_yaml_path = os.path.join("experiment_data", "system_roles", "roles.yaml")
    with open(roles_yaml_path, 'r', encoding='utf-8') as f:
        roles_config = yaml.safe_load(f)

    # 2) Prepare the stiffness matrix processor (for parsing GPT responses)
    stiffness_processor = StiffnessMatrixProcessor(matrices_dir="matrices")

    # 3) Load the ground-truth stiffness matrices for each stage
    gt_stiffness_file = os.path.join("experiment_data", "labels", "ground_truth_stiffness.yaml")
    with open(gt_stiffness_file, 'r', encoding='utf-8') as f:
        gt_stiffness_data = yaml.safe_load(f)

    # Named stages
    stages = [
        "entrance",
        "ytraverse1",
        "corner1",
        "xtraverse",
        "corner2",
        "slant",
        "ytraverse2",
        "exit"
    ]

    # Roles (e.g. ["role1","role2","role3","role4","role5"])
    system_role_keys = list(roles_config.keys())

    # Conversation prior can be 'none', 'home', or 'lab'
    conversation_prior_options = ["none", "home", "lab"]
    # Image resolution options
    resolution_options = ["low", "high"]

    evaluator = StiffnessMatrixEvaluator()
    results = []

    # Number of repeated trials per combination
    num_repetitions = 5  # typical compromise in studies

    for stage in stages:
        if stage not in gt_stiffness_data:
            logging.warning(f"No GT data found for stage='{stage}', skipping.")
            continue
        gt_matrix = gt_stiffness_data[stage]["stiffness_matrix"]

        img_url_base = "https://prompt-engineering-experiment.ngrok.io/images"

        for role_key in system_role_keys:
            role_info = roles_config[role_key]
            system_role_content = role_info["system_role_content"]
            conversation_history_file = role_info["conversation_history_file"]

            for conversation_prior in conversation_prior_options:
                # Select which ground-truth messages to load (or none)
                if conversation_prior == "none":
                    pre_knowledge = None
                elif conversation_prior == "home":
                    gt_path = os.path.join("experiment_data", "labels", "ground_truth_messages_home.json")
                    with open(gt_path, 'r', encoding='utf-8') as f:
                        pre_knowledge = json.load(f)
                elif conversation_prior == "lab":
                    gt_path = os.path.join("experiment_data", "labels", "ground_truth_messages_lab.json")
                    with open(gt_path, 'r', encoding='utf-8') as f:
                        pre_knowledge = json.load(f)
                else:
                    raise ValueError(f"Unknown conversation_prior option: {conversation_prior}")

                for resolution in resolution_options:
                    for repetition in range(num_repetitions):
                        # 1) Reset conversation
                        conv_manager = ConversationHistoryProcessor(
                            system_role_content=system_role_content,
                            conversation_history_file=conversation_history_file,
                            pre_knowledge_messages=pre_knowledge
                        )
                        conv_manager.reset_conversation_history()

                        # 2) Build SpeechProcessor
                        speech_processor = SpeechProcessor(
                            system_role_content=system_role_content,
                            conversation_history_file=conversation_history_file,
                            pre_knowledge_messages=pre_knowledge,
                            image_resolution=resolution
                        )

                        # 3) Construct user prompt
                        user_prompt = (
                            f"Trial {repetition+1}/{num_repetitions}, "
                            f"Stage '{stage}': Please describe the image and compute the stiffness matrix."
                        )
                        image_url = f"{img_url_base}/{stage}.jpg"

                        # 4) Get GPT response
                        gpt_response = speech_processor.get_gpt_response_vlm(
                            transcript=user_prompt,
                            image_url=image_url
                        )

                        if not gpt_response:
                            logging.warning(
                                f"No response for stage='{stage}', role='{role_key}', "
                                f"conversation_prior={conversation_prior}, resolution='{resolution}', trial={repetition+1}"
                            )
                            result_record = {
                                "stage": stage,
                                "role": role_key,
                                "conversation_prior": conversation_prior,
                                "resolution": resolution,
                                "repetition": repetition + 1,
                                "estimated_matrix": None,
                                "metrics": None,
                                "conv_history_file": build_conversation_log_filename(
                                    stage, role_key, conversation_prior, resolution, repetition
                                )
                            }
                            results.append(result_record)
                            save_conversation_history(conversation_history_file,
                                                     result_record["conv_history_file"])
                            continue

                        # 5) Extract the matrix
                        estimated_matrix, matrix_path = stiffness_processor.extract_stiffness_matrix(gpt_response)
                        if not estimated_matrix:
                            logging.warning(
                                f"Could not extract a valid stiffness matrix for stage='{stage}', role='{role_key}', "
                                f"conversation_prior={conversation_prior}, resolution='{resolution}', trial={repetition+1}"
                            )
                            result_record = {
                                "stage": stage,
                                "role": role_key,
                                "conversation_prior": conversation_prior,
                                "resolution": resolution,
                                "repetition": repetition + 1,
                                "estimated_matrix": None,
                                "metrics": None,
                                "conv_history_file": build_conversation_log_filename(
                                    stage, role_key, conversation_prior, resolution, repetition
                                )
                            }
                            results.append(result_record)
                            save_conversation_history(conversation_history_file,
                                                     result_record["conv_history_file"])
                            continue

                        # 6) Evaluate
                        metrics = evaluator.evaluate_stiffness_matrix(gt_matrix, estimated_matrix)
                        logging.info(f"Metrics (stage='{stage}', role='{role_key}', trial={repetition+1}): {metrics}")

                        # 7) Save record
                        result_record = {
                            "stage": stage,
                            "role": role_key,
                            "conversation_prior": conversation_prior,
                            "resolution": resolution,
                            "repetition": repetition + 1,
                            "estimated_matrix": estimated_matrix,
                            "metrics": metrics,
                            "conv_history_file": build_conversation_log_filename(
                                stage, role_key, conversation_prior, resolution, repetition
                            )
                        }
                        results.append(result_record)

                        # 8) Save conversation logs
                        save_conversation_history(conversation_history_file, result_record["conv_history_file"])

    # 9) Save aggregated results
    results_file = "experiment_results.json"
    with open(results_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logging.info(f"All experiments completed. Results saved to '{results_file}'.")


def is_numeric_matrix(stiffness_matrix):
    """
    Returns True if the 'stiffness_matrix' is a 3x3 of purely numeric values.
    Otherwise returns False.
    """
    try:
        arr = np.array(stiffness_matrix, dtype=float)
        if arr.shape == (3, 3):
            return True
        else:
            logging.warning(f"Matrix shape is {arr.shape}, not (3,3).")
            return False
    except (ValueError, TypeError):
        # Means there's a non-numeric item or some placeholder text
        logging.warning("Matrix contains non-numeric elements.")
        return False


def build_conversation_log_filename(stage, role_key, conversation_prior, resolution, repetition):
    """
    Builds a unique conversation log filename based on the trial parameters.
    """
    folder = "conversation_logs"
    os.makedirs(folder, exist_ok=True)
    dest_filename = (
        f"stage_{stage}__role_{role_key}"
        f"__prior_{conversation_prior}__res_{resolution}__trial_{repetition+1}.json"
    )
    return os.path.join(folder, dest_filename)


def save_conversation_history(conversation_history_file, dest_path):
    """
    Reads the conversation history JSON from conversation_history_file
    and saves it to the unique path dest_path.
    """
    try:
        with open(conversation_history_file, "r", encoding="utf-8") as f:
            history_data = json.load(f)
    except FileNotFoundError:
        logging.warning(f"Conversation history file not found: {conversation_history_file}")
        history_data = []
    except json.JSONDecodeError as e:
        logging.warning(f"Error decoding conversation history: {e}")
        history_data = []

    try:
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2)
        logging.info(f"Conversation history saved to '{dest_path}'.")
    except Exception as e:
        logging.error(f"Failed to save conversation history to '{dest_path}': {e}")


if __name__ == "__main__":
    main()
