import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import scipy.stats as stats
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['axes.unicode_minus'] = False



sns.set(style="whitegrid")

# Put your stages in order if you'd like to sort them chronologically
chronological_stages = [
    "entrance",
    "ytraverse1",
    "corner1",
    "xtraverse",
    "corner2",
    "slant"
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

    # Extract the 'correct' field from the metrics dictionary (if present)
    def extract_correct(metrics):
        if isinstance(metrics, dict) and "correct" in metrics:
            return metrics["correct"]
        return False  # or np.nan, depending on your preference
    
    df["correct"] = df["metrics"].apply(extract_correct)
    
    # Convert 'correct' to bool, just to be sure
    df["correct"] = df["correct"].astype(bool)
    
    # Add stage_order for sorting
    def stage_to_order(stage):
        try:
            return chronological_stages.index(stage)
        except ValueError:
            return -1  # unknown stage goes last
    
    df["stage_order"] = df["stage"].apply(stage_to_order)

    # Define the stages to exclude
    excluded_stages = {"corner1", "corner2","slant"}

    # Filter out the corners
    df = df[~df["stage"].isin(excluded_stages)]
    
    # Sort by stage order for chronological plots
    df = df.sort_values("stage_order").reset_index(drop=True)
    
    return df

def plot_trial_counts(df, filename="trial_counts_by_stage.png"):
    """
    Creates a bar plot showing the total number of trials and the number of correct trials for each stage.
    """
    trial_counts = df.groupby("stage")["correct"].agg(["count", "sum"]).reset_index()
    trial_counts.rename(columns={"count": "total_trials", "sum": "correct_trials"}, inplace=True)

    plt.figure(figsize=(10, 6))
    width = 0.4  # Bar width for better separation

    # Plot total trials
    plt.bar(trial_counts["stage"], trial_counts["total_trials"], width=width, label="Total Trials", color="lightgray")

    # Plot correct trials
    plt.bar(trial_counts["stage"], trial_counts["correct_trials"], width=width, label="Correct Trials", color="C0")

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Stage")
    plt.ylabel("Number of Trials")
    plt.title("Total Trials vs Correct Trials by Stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename.replace(".png", ".pdf"), dpi=300, format="pdf")

    plt.close()

    print(f"Saved trial counts plot to '{filename}'.")

def plot_trial_counts_per_combination(df, output_dir="trial_count_plots"):
    """
    Generates a bar plot for each (role, prior, resolution) combination, showing 
    total and correct trials at each stage.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Get all unique combinations of (role, prior, resolution)
    unique_combinations = df.groupby(["role", "prior", "resolution"]).size().reset_index()[["role", "prior", "resolution"]]

    for _, row in unique_combinations.iterrows():
        role, prior, resolution = row["role"], row["prior"], row["resolution"]

        # Filter the DataFrame for this specific combination
        df_filtered = df[
            (df["role"] == role) & 
            (df["prior"] == prior) & 
            (df["resolution"] == resolution)
        ]

        if df_filtered.empty:
            print(f"No data found for {role} | {prior} | {resolution}. Skipping plot.")
            continue

        # Group by stage and count total and correct trials
        trial_counts = df_filtered.groupby("stage")["correct"].agg(["count", "sum"]).reset_index()
        trial_counts.rename(columns={"count": "total_trials", "sum": "correct_trials"}, inplace=True)

        # Generate filename dynamically
        filename = f"{output_dir}/trial_counts_{role}_{prior}_{resolution}.png"

        plt.figure(figsize=(10, 6))
        width = 0.4  # Bar width for better separation

        # Plot total trials
        plt.bar(trial_counts["stage"], trial_counts["total_trials"], width=width, label="Total Trials", color="lightgray")

        # Plot correct trials
        plt.bar(trial_counts["stage"], trial_counts["correct_trials"], width=width, label="Correct Trials", color="C0")

        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Stage")
        plt.ylabel("Number of Trials")
        plt.title(f"Total vs Correct Trials ({role} | {prior} | {resolution})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename.replace(".png", ".pdf"), dpi=300, format="pdf")
        plt.close()

        print(f"Saved plot for {role} | {prior} | {resolution} to '{filename}'.")

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
    plt.savefig(filename.replace(".png", ".pdf"), dpi=300, format="pdf")
    plt.close()
    
    print(f"Saved plot to '{filename}'.")

def plot_accuracy_nested_bar(df, filename="accuracy_by_stage_role_prior_resolution.png"):
    """
    Creates a grouped bar plot with:
    - stage on the x-axis
    - each combination of (role, prior, resolution) as the hue
    
    This way, at each stage, you see multiple bars split by role -> prior -> resolution.
    """
    required_cols = {"stage", "role", "prior", "resolution", "correct"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Skipping nested bar plot because these columns are missing: {missing}")
        return

    # Print total trials to verify
    total_trials = len(df)
    print(f"Total number of trials: {total_trials}")
    
    if total_trials != 540:
        print(f"Warning: Expected 540 trials but found {total_trials}")

    # Print unique values in key columns to check data consistency
    print("Unique values in key columns:")
    for col in ["stage", "role", "prior", "resolution"]:
        unique_vals = df[col].unique()
        print(f"  {col}: {unique_vals}")

    # Group to get mean accuracy by stage, role, prior, resolution
    grouped = (
        df.groupby(["stage", "role", "prior", "resolution"])["correct"]
        .mean()
        .reset_index(name="accuracy")
    )
    
    # Print accuracy statistics
    print("\nMean accuracy per configuration:")
    print(grouped)

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
    plt.savefig(filename.replace(".png", ".pdf"), dpi=300, format="pdf")
    plt.close()
    
    print(f"Saved nested bar plot to '{filename}'.")

def plot_accuracy_nested_bar_filtered(df, filename="accuracy_by_stage_role_prior_resolution_filtered.png"):
    """
    Creates a grouped bar plot for accuracy by stage, but only for the 
    role-prior-resolution combination: role3 | none | high.
    """
    required_cols = {"stage", "role", "prior", "resolution"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Skipping filtered bar plot because these columns are missing: {missing}")
        return
    
    # Filter for the specific combination
    df_filtered = df[
        (df["role"] == "role3") & 
        (df["prior"] == "none") & 
        (df["resolution"] == "high")
    ]

    if df_filtered.empty:
        print("No data found for role3 | none | high. Skipping filtered plot.")
        return

    # Group to get mean accuracy by stage
    grouped = df_filtered.groupby("stage")["correct"].mean().reset_index(name="accuracy")

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x="stage", 
        y="accuracy",
        data=grouped,
        color="C0"
    )
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Stage")
    plt.ylabel("Fraction Correct")
    plt.title("Accuracy by Stage (Only Role3 | None | High)")
    plt.tight_layout()
    plt.savefig(filename.replace(".png", ".pdf"), dpi=300, format="pdf")
    plt.close()
    print(f"Saved filtered bar plot to '{filename}'.")


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
    plt.savefig(filename.replace(".png", ".pdf"), dpi=300, format="pdf")
    plt.close()
    print(f"Saved overall (role, prior, resolution) plot to '{filename}'.")

def calculate_accuracy_per_combination(df):
    """
    Computes and prints the overall accuracy for each (role, prior, resolution) combination.
    Returns a DataFrame sorted in descending order of accuracy.
    """
    required_cols = {"role", "prior", "resolution", "correct"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Skipping accuracy calculation because these columns are missing: {missing}")
        return pd.DataFrame()  # Return an empty DataFrame if required columns are missing
    
    # Group by (role, prior, resolution) and calculate accuracy
    accuracy_df = (
        df.groupby(["role", "prior", "resolution"])["correct"]
        .agg(["count", "sum"])  # Count total trials and correct trials
        .reset_index()
    )
    
    # Compute accuracy (correct / total)
    accuracy_df["accuracy"] = accuracy_df["sum"] / accuracy_df["count"]

    # Rename columns for clarity
    accuracy_df.rename(columns={"count": "total_trials", "sum": "correct_trials"}, inplace=True)

    # Sort by accuracy in descending order
    accuracy_df = accuracy_df.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

    # Print results
    print("\n=== Ranked Accuracy by (Role, Prior, Resolution) ===")
    print(accuracy_df)

    return accuracy_df

def save_ranked_accuracy_as_tuples(df, filename="ranked_accuracy.py"):
    """
    Computes overall accuracy for each (role, prior, resolution) combination,
    sorts it in descending order, formats it as a list of tuples,
    prints it, and saves it to a Python file.
    """
    required_cols = {"role", "prior", "resolution", "correct"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Skipping accuracy calculation because these columns are missing: {missing}")
        return
    
    # Group by (role, prior, resolution) and calculate accuracy
    accuracy_df = (
        df.groupby(["role", "prior", "resolution"])["correct"]
        .agg(["count", "sum"])  # Count total trials and correct trials
        .reset_index()
    )
    
    # Compute accuracy (correct / total)
    accuracy_df["accuracy"] = accuracy_df["sum"] / accuracy_df["count"]

    # Sort by accuracy in descending order
    accuracy_df = accuracy_df.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

    # Convert to a list of tuples
    accuracy_tuples = [
        (row["role"], row["prior"], row["resolution"], round(row["accuracy"], 6)) 
        for _, row in accuracy_df.iterrows()
    ]

    # Print formatted output
    print("\nsave_ranked_accuracy_as_tuples=== Ranked Accuracy (Tuple Format) ===")
    print("data = [")
    for item in accuracy_tuples:
        print(f"    {item},")
    print("]")

    # Save to a Python file
    with open(filename, "w") as f:
        f.write("data = [\n")
        for item in accuracy_tuples:
            f.write(f"    {item},\n")
        f.write("]\n")

    print(f"Ranked accuracy saved to '{filename}'.")

    return accuracy_tuples

def analyze_accuracy_with_ci(data, n_trials=30, save_plot="accuracy_confidence_intervals.png"):
    """
    Analyzes accuracy of (role, prior, resolution) combinations with 95% confidence intervals (CI).
    Identifies the best configuration and determines statistical significance by checking CI overlap.

    Parameters:
    - data: List of tuples (role, prior, resolution, accuracy)
    - n_trials: Number of trials per configuration (default: 30)
    - save_plot: Filename to save the plot

    Returns:
    - DataFrame containing accuracy, CI, and statistical comparison results.
    """

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["role", "prior", "resolution", "accuracy"])

    # Compute 95% CI for each accuracy value
    def compute_ci(accuracy, n):
        """Computes 95% confidence interval for a proportion"""
        z = 1.96  # 95% confidence level
        se = np.sqrt((accuracy * (1 - accuracy)) / n)  # Standard error
        ci_lower = accuracy - z * se
        ci_upper = accuracy + z * se
        return round(ci_lower, 3), round(ci_upper, 3)

    df["CI"] = df["accuracy"].apply(lambda acc: compute_ci(acc, n_trials))

    # Sort by accuracy (best first)
    df = df.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

    # Identify best-performing configuration
    best_accuracy = df.iloc[0]["accuracy"]
    best_CI = df.iloc[0]["CI"]

    # Check CI overlap with all other configurations
    overlapping = []
    non_overlapping = []

    for i in range(1, len(df)):  # Compare with all others
        curr_accuracy = df.iloc[i]["accuracy"]
        curr_CI = df.iloc[i]["CI"]

        # Check if CI ranges overlap
        if (best_CI[0] <= curr_CI[1]) and (curr_CI[0] <= best_CI[1]):
            overlapping.append((df.iloc[i][["role", "prior", "resolution"]].tolist(), curr_accuracy, curr_CI))
        else:
            non_overlapping.append((df.iloc[i][["role", "prior", "resolution"]].tolist(), curr_accuracy, curr_CI))

    # Print results
    print(f"\n=== Best Configuration ===")
    print(f"{df.iloc[0][['role', 'prior', 'resolution']].tolist()} - Accuracy: {best_accuracy}, CI: {best_CI}\n")

    print("Configurations with **Overlapping CI** (Not Statistically Significant Difference):")
    for entry in overlapping:
        print(f"- {entry[0]}: Accuracy = {entry[1]}, CI = {entry[2]}")

    print("\nConfigurations with **Non-Overlapping CI** (Statistically Significant Difference!):")
    for entry in non_overlapping:
        print(f"- {entry[0]}: Accuracy = {entry[1]}, CI = {entry[2]}")

    # Final conclusion
    if len(non_overlapping) > 0:
        print("\n✅ The best configuration is **statistically significantly better** than at least some other configurations!")
    else:
        print("\n❌ The confidence intervals overlap with all configurations. No statistically significant difference.")

    # Visualization
    plt.figure(figsize=(8, 6))

    # Plot accuracy values with error bars (showing confidence intervals)
    for i in range(len(df)):
        accuracy = df.iloc[i]["accuracy"]
        ci = df.iloc[i]["CI"]
        role = df.iloc[i]["role"]
        prior = df.iloc[i]["prior"]
        resolution = df.iloc[i]["resolution"]

        # Define color: Red for the best, blue for the rest
        color = 'red' if i == 0 else 'blue'

        plt.errorbar(
            accuracy, i, 
            xerr=[[accuracy - ci[0]], [ci[1] - accuracy]], 
            fmt='o', color=color, capsize=5, label="_nolegend_" if i > 0 else "Best Configuration"
        )

        # Annotate each point directly on top of the interval line
        plt.text(accuracy, i + 0.2, f"{role}, {prior}, {resolution}", 
                 verticalalignment='bottom', horizontalalignment='center')

    # Best CI range lines
    plt.axvline(x=best_CI[0], color='red', linestyle='--', linewidth=1, label="Best CI Range")
    plt.axvline(x=best_CI[1], color='red', linestyle='--', linewidth=1)

    # Formatting
    plt.xlabel("Accuracy")
    plt.ylabel("Configuration Ranking")
    plt.title("Confidence Intervals of Accuracy for Each Configuration")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    # Save the plot
    plt.savefig(save_plot.replace(".png", ".pdf"), dpi=300, format="pdf")
    plt.close()

    print(f"\n📊 Confidence interval plot saved as '{save_plot}'.")

    return df

def main():
    json_file = "experiment_results_renamed_home_ideal.json"
    
    # Load and process data
    df = load_and_process_data(json_file)
    
    accuracy_df = calculate_accuracy_per_combination(df)
    data=save_ranked_accuracy_as_tuples(df)
    # Print accuracy summary
    print("\n=== Accuracy Summary ===")
    summarize_accuracy(df)
    
    # Generate all plots
    plot_accuracy_by_group(df, ["stage"], filename="accuracy_by_stage.png")
    plot_accuracy_by_group(df, ["role"], filename="accuracy_by_role.png")
    plot_accuracy_by_group(df, ["stage", "role"], filename="accuracy_by_stage_role.png")
    plot_trial_counts(df, filename="trial_counts_by_stage.png")
    plot_trial_counts_per_combination(df, output_dir="trial_count_plots")
    # Full nested plot (all roles, priors, resolutions)
    plot_accuracy_nested_bar(df, filename="accuracy_by_stage_role_prior_resolution.png")
    
    confidence_intervals = analyze_accuracy_with_ci(data, n_trials=30, save_plot="accuracy_confidence_intervals.png")

    # Filtered plot (only role3 | none | high)
    plot_accuracy_nested_bar_filtered(df, filename="accuracy_by_stage_role_prior_resolution_filtered.png")
    
    # Overall accuracy for each role-prior-resolution combination
    plot_accuracy_by_role_prior_resolution_overall(df, filename="accuracy_by_role_prior_resolution_overall.png")

if __name__ == "__main__":
    main()

