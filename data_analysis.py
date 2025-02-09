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
    """
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found.")
        sys.exit(1)
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Drop rows that have no matrix or metrics
    df = df.dropna(subset=["estimated_matrix", "metrics"]).reset_index(drop=True)
    
    # Extract the 'correct' field from the metrics dictionary (if present)
    def extract_correct(metrics):
        # metrics is expected to be a dict with a 'correct' key, e.g. {"correct": True}
        if isinstance(metrics, dict) and "correct" in metrics:
            return metrics["correct"]
        return False  # or np.nan, depending on your preference
    
    df["correct"] = df["metrics"].apply(extract_correct)
    
    # Filter down to rows where we indeed have a True/False in 'correct'
    df = df.dropna(subset=["correct"]).reset_index(drop=True)
    
    # Convert 'correct' to bool, just to be sure
    df["correct"] = df["correct"].astype(bool)
    
    # Optionally add stage_order for sorting
    def stage_to_order(stage):
        try:
            return chronological_stages.index(stage)
        except ValueError:
            return -1  # unknown stage goes last
    
    df["stage_order"] = df["stage"].apply(stage_to_order)
    
    # Sort by stage order if you want chronological plots
    df = df.sort_values("stage_order").reset_index(drop=True)
    
    return df


def summarize_accuracy(df):
    """
    Prints overall accuracy, plus a breakdown by stage or any other factors.
    """
    total_trials = len(df)
    total_correct = df["correct"].sum()
    accuracy = total_correct / total_trials if total_trials > 0 else 0.0
    
    print(f"Total Trials: {total_trials}")
    print(f"Total Correct: {total_correct}")
    print(f"Overall Accuracy: {accuracy:.2f}")
    
    # If you want a breakdown by stage:
    acc_by_stage = df.groupby("stage")["correct"].mean().reset_index()
    print("\nAccuracy by Stage:")
    print(acc_by_stage)


def plot_accuracy_by_group(df, group_cols, filename="accuracy_by_group.png"):
    """
    Creates a bar plot of correctness (fraction correct) grouped by the given columns.
    For example, group_cols=["stage"], or multiple columns like ["stage","role"].
    """
    # Group by specified columns and compute mean correctness
    group_stats = df.groupby(group_cols)["correct"].mean().reset_index()
    
    # For a simpler label on the x-axis, combine columns if you have multiple
    if len(group_cols) > 1:
        group_stats["group_label"] = group_stats[group_cols].astype(str).agg(" | ".join, axis=1)
    else:
        group_stats["group_label"] = group_stats[group_cols[0]]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="group_label", y="correct", data=group_stats, palette="Blues_d")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("Group")
    plt.ylabel("Fraction Correct")
    plt.title(f"Accuracy by {', '.join(group_cols)}")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Saved plot to '{filename}'.")


def plot_accuracy_across_stages_all_factors(df, filename="accuracy_all_factors.png"):
    """
    Creates a single line plot showing accuracy across stages for each combination
    of role, prior, and resolution. This can get busy if you have many combinations.
    """
    # Confirm you have columns named 'role', 'prior', 'resolution' in your dataset.
    # If your dataset uses different column names, update them here.
    # We'll assume 'role', 'prior', and 'resolution' exist.
    
    # Calculate mean accuracy by stage order + other factors
    group_cols = ["stage_order", "stage", "role", "prior", "resolution"]
    grouped = df.groupby(group_cols)["correct"].mean().reset_index()
    
    # Create a single label that combines all factors except stage/stage_order
    # e.g. "roleA | prior=conv | res=HD"
    grouped["combo_label"] = (
        grouped["role"].astype(str)
        + " | prior="
        + grouped["prior"].astype(str)
        + " | res="
        + grouped["resolution"].astype(str)
    )
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        x="stage_order",
        y="correct",
        hue="combo_label",
        data=grouped,
        marker="o",
        legend="full"
    )
    
    # Replace stage_order tick labels with the actual stage names in chronological order
    unique_stage_orders = sorted(df["stage_order"].unique())
    stage_labels = [df.loc[df["stage_order"] == so, "stage"].iloc[0] for so in unique_stage_orders]
    plt.xticks(unique_stage_orders, stage_labels, rotation=45, ha="right")
    
    plt.ylim(0, 1)
    plt.xlabel("Stage")
    plt.ylabel("Fraction Correct")
    plt.title("Accuracy Across Stages (Role × Prior × Resolution)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Saved plot to '{filename}'.")


def plot_accuracy_facet_grid(df, row_var, col_var, filename="facet_accuracy.png"):
    """
    Creates a FacetGrid of barplots for accuracy, allowing quick comparisons across
    two categorical variables. For instance:
      - row_var = "resolution"
      - col_var = "prior"
    Then color/hue can be "role" or vice versa. Adjust to your needs.
    """
    # We'll group by stage and the chosen factors to get average accuracy
    group_cols = ["stage", row_var, col_var, "role"]  # 'role' is the hue
    grouped = df.groupby(group_cols)["correct"].mean().reset_index()
    
    # Create the facet grid
    g = sns.FacetGrid(
        grouped, 
        row=row_var, 
        col=col_var, 
        margin_titles=True, 
        sharey=True,
        height=4, 
        aspect=1.2
    )
    # Within each facet, plot a bar chart with hue="role"
    g.map_dataframe(
        sns.barplot, 
        x="stage", 
        y="correct", 
        hue="role", 
        palette="muted",
        errorbar=None
    )
    
    # Rotate x labels, set y limit to [0,1], and adjust layout
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Stage")
        ax.set_ylabel("Accuracy")
    
    g.add_legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Saved facet grid plot to '{filename}'.")


def main():
    json_file = "experiment_results.json"
    
    # 1) Load the data
    df = load_and_process_data(json_file)
    
    # 2) Print basic accuracy summary
    print("\n=== Accuracy Summary ===")
    summarize_accuracy(df)
    
    # 3) Plot accuracy by stage (bar plot)
    plot_accuracy_by_group(df, ["stage"], filename="accuracy_by_stage.png")
    
    # 4) Plot accuracy by role (bar plot)
    plot_accuracy_by_group(df, ["role"], filename="accuracy_by_role.png")
    
    # 5) Plot accuracy by stage & role together (bar plot)
    plot_accuracy_by_group(df, ["stage", "role"], filename="accuracy_by_stage_role.png")
    
    # 6) Single line plot: accuracy across stages for all factors (role, prior, resolution)
    #    This might be very busy if you have many combinations
    if all(col in df.columns for col in ["role", "prior", "resolution"]):
        plot_accuracy_across_stages_all_factors(df, filename="accuracy_all_factors.png")
    else:
        print("Skipping multi-factor line plot because 'role', 'prior', or 'resolution' is missing.")
    
    # 7) Optional: Facet grid to compare accuracy by resolution/prior/role
    #    Adjust row_var, col_var as needed
    if all(col in df.columns for col in ["role", "prior", "resolution"]):
        plot_accuracy_facet_grid(
            df, 
            row_var="resolution", 
            col_var="prior",
            filename="facet_accuracy_resolution_prior.png"
        )
    else:
        print("Skipping facet grid because 'role', 'prior', or 'resolution' is missing.")


if __name__ == "__main__":
    main()
