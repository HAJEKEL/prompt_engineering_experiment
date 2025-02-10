import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sns.set(style="whitegrid")

# Put your stages in order if you'd like to sort them chronologically
chronological_stages = [
    "entrance",
    "ytraverse1",
    "corner1",
    "xtraverse",
    "corner2",
    "slant",
    "ytraverse2",
    "exit"
]

def load_and_process_data(json_file):
    """
    Load the JSON data into a pandas DataFrame.
    Keep only rows that have an 'estimated_matrix' and a 'metrics' dict with a boolean 'correct'.
    Create a boolean column 'correct' from that dict.
    Rename 'conversation_prior' -> 'prior' so we can do multi-factor plots.
    """
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found.")
        sys.exit(1)
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Rename conversation_prior to prior so we can standardize the naming
    if "conversation_prior" in df.columns:
        df.rename(columns={"conversation_prior": "prior"}, inplace=True)

    # Drop rows that have no matrix or metrics
    df = df.dropna(subset=["estimated_matrix", "metrics"]).reset_index(drop=True)
    
    # Extract the 'correct' field from the metrics dictionary (if present)
    def extract_correct(metrics):
        if isinstance(metrics, dict) and "correct" in metrics:
            return metrics["correct"]
        return False  # or np.nan, depending on your preference
    
    df["correct"] = df["metrics"].apply(extract_correct)
    
    # Filter down to rows where 'correct' is True/False
    df = df.dropna(subset=["correct"]).reset_index(drop=True)
    
    # Convert 'correct' to bool, just to be sure
    df["correct"] = df["correct"].astype(bool)
    
    # Add stage_order for sorting
    def stage_to_order(stage):
        try:
            return chronological_stages.index(stage)
        except ValueError:
            return -1  # unknown stage goes last
    
    df["stage_order"] = df["stage"].apply(stage_to_order)
    
    # Sort by stage order for chronological plots
    df = df.sort_values("stage_order").reset_index(drop=True)
    
    return df

def summarize_accuracy(df):
    """
    Prints overall accuracy, plus a breakdown by stage.
    """
    total_trials = len(df)
    total_correct = df["correct"].sum()
    accuracy = total_correct / total_trials if total_trials > 0 else 0.0
    
    print(f"Total Trials: {total_trials}")
    print(f"Total Correct: {total_correct}")
    print(f"Overall Accuracy: {accuracy:.2f}")
    
    acc_by_stage = df.groupby("stage")["correct"].mean().reset_index()
    print("\nAccuracy by Stage:")
    print(acc_by_stage)

def plot_accuracy_by_group(df, group_cols, filename="accuracy_by_group.png"):
    """
    Creates a single-level bar plot of correctness (fraction correct) grouped by columns in group_cols.
    """
    group_stats = df.groupby(group_cols)["correct"].mean().reset_index()
    
    # Combine columns for x-axis labels if you have more than one grouping
    if len(group_cols) > 1:
        group_stats["group_label"] = group_stats[group_cols].astype(str).agg(" | ".join, axis=1)
    else:
        group_stats["group_label"] = group_stats[group_cols[0]]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="group_label", 
        y="correct", 
        data=group_stats, 
        color="C0"
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("Group")
    plt.ylabel("Fraction Correct")
    plt.title(f"Accuracy by {', '.join(group_cols)}")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Saved plot to '{filename}'.")

def plot_accuracy_nested_bar(df, filename="accuracy_by_stage_role_prior_resolution.png"):
    """
    Creates a grouped bar plot with:
    - stage on the x-axis
    - each combination of (role, prior, resolution) as the hue
    
    This way, at each stage, you see multiple bars split by role -> prior -> resolution.
    """
    required_cols = {"stage", "role", "prior", "resolution"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Skipping nested bar plot because these columns are missing: {missing}")
        return
    
    # Group to get mean accuracy by stage, role, prior, resolution
    grouped = (
        df.groupby(["stage", "role", "prior", "resolution"])["correct"]
        .mean()
        .reset_index(name="accuracy")
    )
    
    # Create a single label for the hue (role|prior|resolution)
    grouped["combo_label"] = (
        grouped["role"].astype(str)
        + " | "
        + grouped["prior"].astype(str)
        + " | "
        + grouped["resolution"].astype(str)
    )
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x="stage", 
        y="accuracy", 
        hue="combo_label", 
        data=grouped,
        palette="tab20"
    )
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Stage")
    plt.ylabel("Fraction Correct")
    plt.title("Accuracy by Stage, Role, Prior, Resolution")
    plt.legend(title="Role | Prior | Resolution", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved nested bar plot to '{filename}'.")

def plot_accuracy_by_role_prior_resolution_overall(df, filename="accuracy_by_role_prior_resolution_overall.png"):
    """
    Creates a bar plot showing the overall accuracy (across all stages) for each
    combination of (role, prior, resolution).
    """
    required_cols = {"role", "prior", "resolution"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Skipping overall bar plot because these columns are missing: {missing}")
        return
    
    # Group by (role, prior, resolution) to get mean accuracy across all stages
    grouped = (
        df.groupby(["role", "prior", "resolution"])["correct"]
        .mean()
        .reset_index(name="accuracy")
    )

    # Create a single label for the x-axis (role|prior|resolution)
    grouped["combo_label"] = (
        grouped["role"].astype(str)
        + " | "
        + grouped["prior"].astype(str)
        + " | "
        + grouped["resolution"].astype(str)
    )
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x="combo_label",
        y="accuracy",
        data=grouped,
        palette="tab20"
    )
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Role | Prior | Resolution")
    plt.ylabel("Fraction Correct")
    plt.title("Overall Accuracy by (Role, Prior, Resolution)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved overall (role, prior, resolution) plot to '{filename}'.")

def rank_role_prior_resolution(df):
    """
    Computes a ranking of all (role, prior, resolution) combinations by
    their overall accuracy across all stages (descending).
    Prints a table of the results for easy reference.
    """
    required_cols = {"role", "prior", "resolution"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Skipping ranking because these columns are missing: {missing}")
        return None
    
    # Group by (role, prior, resolution) to get mean accuracy across all stages
    grouped = (
        df.groupby(["role", "prior", "resolution"])["correct"]
        .mean()
        .reset_index(name="accuracy")
    )
    
    # Sort in descending order of accuracy
    grouped = grouped.sort_values(by="accuracy", ascending=False).reset_index(drop=True)
    
    print("\n=== Ranking of (role, prior, resolution) by Overall Accuracy ===")
    print(grouped)
    
    return grouped

def main():
    json_file = "experiment_results.json"
    
    # 1) Load the data
    df = load_and_process_data(json_file)
    
    # 2) Print basic accuracy summary
    print("\n=== Accuracy Summary ===")
    summarize_accuracy(df)
    
    # 3) Plot some simpler bar charts
    plot_accuracy_by_group(df, ["stage"], filename="accuracy_by_stage.png")
    plot_accuracy_by_group(df, ["role"], filename="accuracy_by_role.png")
    plot_accuracy_by_group(df, ["stage", "role"], filename="accuracy_by_stage_role.png")
    
    # 4) The existing nested bar plot for stage × role × prior × resolution
    plot_accuracy_nested_bar(df, filename="accuracy_by_stage_role_prior_resolution.png")
    
    # 5) Overall accuracy plot by role, prior, resolution (all stages combined)
    plot_accuracy_by_role_prior_resolution_overall(df, filename="accuracy_by_role_prior_resolution_overall.png")
    
    # 6) NEW: Print a ranking table of (role, prior, resolution) combos by overall accuracy
    rank_role_prior_resolution(df)

if __name__ == "__main__":
    main()

