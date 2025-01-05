#!/usr/bin/env python3
import os
import json
import yaml
import logging
import numpy as np
from pathlib import Path

from functions.conversation_history_processor import ConversationHistoryProcessor
from functions.speech_processor import SpeechProcessor
from functions.stiffness_matrix_processor import StiffnessMatrixProcessor
from functions.stiffness_matrix_evaluator import StiffnessMatrixEvaluator

def main():
    logging.basicConfig(level=logging.INFO)

    # 1) Load roles from YAML
    roles_yaml_path = os.path.join("experiment_data", "system_roles", "roles.yaml")
    with open(roles_yaml_path, 'r', encoding='utf-8') as f:
        roles_config = yaml.safe_load(f)

    # 2) Prepare the stiffness matrix processor (for parsing GPT responses)
    stiffness_processor = StiffnessMatrixProcessor(matrices_dir="matrices")

    # 3) Load ground-truth conversation primer
    ground_truth_file = os.path.join("experiment_data", "labels", "ground_truth_messages.json")
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth_messages = json.load(f)

    # 4) Load the ground-truth stiffness matrices for each stage
    gt_stiffness_file = os.path.join("experiment_data", "labels", "ground_truth_stiffness.yaml")
    with open(gt_stiffness_file, 'r', encoding='utf-8') as f:
        gt_stiffness_data = yaml.safe_load(f)
        # E.g. gt_stiffness_data["entrance"] = [[100,0,0],[0,100,0],[0,0,100]], etc.

    # Instead of range(1..9), define descriptive stage names matching your YAML keys:
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

    # Updated roles (assuming you removed "role0")
    # E.g. ["role1", "role2", "role3", "role4", "role5"]
    system_role_keys = list(roles_config.keys())

    conversation_prior_options = [False, True]
    resolution_options = ["low", "high"]

    evaluator = StiffnessMatrixEvaluator()
    results = []

    # Nested loops
    for stage in stages:
        # Retrieve ground truth matrix
        if stage not in gt_stiffness_data:
            logging.warning(f"No GT data found for stage={stage}, skipping.")
            continue
        gt_matrix = gt_stiffness_data[stage]["stiffness_matrix"]

        # Where are the images hosted?
        img_url_base = "https://prompt-engineering-experiment.ngrok.io/images"

        for role_key in system_role_keys:
            role_info = roles_config[role_key]
            system_role_content = role_info["system_role_content"]
            conversation_history_file = role_info["conversation_history_file"]

            for use_conv_prior in conversation_prior_options:
                this_pre_knowledge = ground_truth_messages if use_conv_prior else None

                for resolution in resolution_options:
                    # 1) Reset conversation
                    conv_manager = ConversationHistoryProcessor(
                        system_role_content=system_role_content,
                        conversation_history_file=conversation_history_file,
                        pre_knowledge_messages=this_pre_knowledge
                    )
                    conv_manager.reset_conversation_history()

                    # 2) Build SpeechProcessor
                    speech_processor = SpeechProcessor(
                        system_role_content=system_role_content,
                        conversation_history_file=conversation_history_file,
                        pre_knowledge_messages=this_pre_knowledge,
                        image_resolution=resolution
                    )

                    # 3) Construct user prompt
                    user_prompt = f"Stage {stage}: Please describe the image and compute the stiffness matrix."

                    # 4) Build the image URL (must match how you named each JPG)
                    # e.g. "entrance.jpg", "ytraverse1.jpg", etc.
                    image_url = f"{img_url_base}/{stage}.jpg"

                    # 5) Get GPT response
                    gpt_response = speech_processor.get_gpt_response_vlm(
                        transcript=user_prompt,
                        image_url=image_url
                    )

                    if not gpt_response:
                        logging.warning(f"No response for stage={stage}, role={role_key}, "
                                        f"conv_prior={use_conv_prior}, resolution={resolution}")
                        results.append({
                            "stage": stage,
                            "role": role_key,
                            "use_conv_prior": use_conv_prior,
                            "resolution": resolution,
                            "estimated_matrix": None,
                            "metrics": None
                        })
                        continue

                    # 6) Parse out the stiffness matrix
                    estimated_matrix, matrix_path = stiffness_processor.extract_stiffness_matrix(gpt_response)
                    if not estimated_matrix:
                        logging.warning(f"Could not extract matrix for stage={stage}, role={role_key}, "
                                        f"conv_prior={use_conv_prior}, resolution={resolution}")
                        results.append({
                            "stage": stage,
                            "role": role_key,
                            "use_conv_prior": use_conv_prior,
                            "resolution": resolution,
                            "estimated_matrix": None,
                            "metrics": None
                        })
                        continue

                    # 7) Evaluate
                    metrics = evaluator.evaluate_stiffness_matrix(gt_matrix, estimated_matrix)
                    logging.info(f"Metrics for stage={stage}, role={role_key}, "
                                 f"conv_prior={use_conv_prior}, resolution={resolution}:\n{metrics}")

                    # 8) Save record
                    results.append({
                        "stage": stage,
                        "role": role_key,
                        "use_conv_prior": use_conv_prior,
                        "resolution": resolution,
                        "estimated_matrix": estimated_matrix,
                        "metrics": metrics
                    })

    # 9) Save results
    results_file = "experiment_results.json"
    with open(results_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logging.info(f"All experiments completed. Results saved to {results_file}")


if __name__ == "__main__":
    main()
