import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Define the chronological order of stages
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
    Load the JSON data and process it into a pandas DataFrame.
    Filters out failed trials where 'estimated_matrix' or 'metrics' are null.
    Further filters out trials with incomplete 'metrics' fields.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(data)
    
    # Initial filter: Remove trials where 'estimated_matrix' or 'metrics' are null
    df_success = df.dropna(subset=['estimated_matrix', 'metrics']).reset_index(drop=True)
    initial_success_count = len(df_success)
    print(f"Initial successful trials (after dropping null 'estimated_matrix' or 'metrics'): {initial_success_count}")
    
    # Define required metric keys
    required_metric_keys = ['mae', 'mse', 'frobenius_norm_diff', 'angular_deviation', 'eigenvalue_magnitude_diff']
    
    # Function to check if all required keys are present in 'metrics'
    def has_all_required_metrics(metrics):
        if not isinstance(metrics, dict):
            return False
        return all(key in metrics for key in required_metric_keys)
    
    # Further filter: Keep only trials with all required metric keys
    df_complete_metrics = df_success[df_success['metrics'].apply(has_all_required_metrics)].reset_index(drop=True)
    complete_metrics_count = len(df_complete_metrics)
    incomplete_metrics_count = initial_success_count - complete_metrics_count
    print(f"Trials with complete 'metrics': {complete_metrics_count}")
    print(f"Trials excluded due to incomplete 'metrics': {incomplete_metrics_count}")
    
    if incomplete_metrics_count > 0:
        print("Note: Some trials were excluded because their 'metrics' fields were incomplete.")
    
    df_success = df_complete_metrics.copy()
    
    # Normalize the 'metrics' column
    metrics_df = pd.json_normalize(df_success['metrics'])
    
    # Handle 'angular_deviation' and 'eigenvalue_magnitude_diff' which are lists
    # Convert lists to separate columns
    for metric_list, prefix in [('angular_deviation', 'angular_deviation_'),
                                ('eigenvalue_magnitude_diff', 'eigenvalue_diff_')]:
        if metric_list in metrics_df.columns:
            # Check if the list has the expected length
            expected_length = 3  # Based on the snippet you provided
            actual_lengths = metrics_df[metric_list].apply(lambda x: len(x) if isinstance(x, list) else 0)
            max_length = actual_lengths.max()
            if max_length != expected_length:
                print(f"Warning: '{metric_list}' has varying lengths. Expected {expected_length}, found up to {max_length}.")
                # Adjust expected_length based on actual data
                expected_length = max_length
            # Expand the list into separate columns
            expanded_df = pd.DataFrame(metrics_df[metric_list].tolist(),
                                       columns=[f"{prefix}{i+1}" for i in range(expected_length)])
            metrics_df = pd.concat([metrics_df.drop(metric_list, axis=1), expanded_df], axis=1)
        else:
            # If the key is missing, fill with NaNs
            print(f"Warning: '{metric_list}' key is missing in some 'metrics'. Filling with NaNs.")
            metrics_df[[f"{prefix}{i+1}" for i in range(3)]] = np.nan
    
    # Concatenate the metrics back to the main DataFrame
    df_success = pd.concat([df_success, metrics_df], axis=1)
    
    # Add stage order based on chronological_stages
    df_success['stage_order'] = df_success['stage'].apply(
        lambda x: chronological_stages.index(x) if x in chronological_stages else -1
    )
    
    # Optional: Sort by stage_order for chronological plotting
    df_success = df_success.sort_values(by='stage_order').reset_index(drop=True)
    
    return df_success

def perform_statistical_analysis(df, response_variable='mae'):
    """
    Perform ANOVA to determine the significance of each factor on the response variable.
    """
    # Define the formula for ANOVA
    formula = f"{response_variable} ~ C(stage) + C(role) + C(use_conv_prior) + C(resolution)"
    
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    return anova_table

def calculate_variance(df, group_vars, response_variable='mae'):
    """
    Calculate the variance of the response variable for each combination of group_vars.
    """
    variance_df = df.groupby(group_vars)[response_variable].var().reset_index()
    variance_df = variance_df.rename(columns={response_variable: 'variance'})
    return variance_df

def plot_boxplots(df, x_var, y_var, hue_var, title, xlabel, ylabel, filename):
    """
    Create and save a boxplot.
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=x_var, y=y_var, hue=hue_var, data=df)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(title=hue_var, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_variance(variance_df, group_vars, filename):
    """
    Create and save a bar plot for variance.
    """
    plt.figure(figsize=(14, 8))
    # Create a combined group label
    variance_df['group'] = variance_df[group_vars].astype(str).agg(' | '.join, axis=1)
    sns.barplot(x='group', y='variance', data=variance_df, palette='viridis')
    plt.title('Variance of MAE by Variable Combinations', fontsize=16)
    plt.xlabel('Variable Combinations', fontsize=14)
    plt.ylabel('Variance of MAE', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_metrics_across_stages(df, stage_order, response_variable='mae', filename='metrics_across_stages.png'):
    """
    Plot the response variable across stages in chronological order.
    """
    plt.figure(figsize=(14, 8))
    sns.lineplot(x='stage_order', y=response_variable, hue='role', data=df, marker='o')
    plt.xticks(ticks=range(len(stage_order)), labels=stage_order, rotation=45)
    plt.title(f'{response_variable.upper()} Across Stages', fontsize=16)
    plt.xlabel('Stage', fontsize=14)
    plt.ylabel(response_variable.upper(), fontsize=14)
    plt.legend(title='Role', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    # Path to the JSON results file
    json_file = 'experiment_results.json'
    
    # Check if the file exists
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found. Please ensure the file is in the correct directory.")
        sys.exit(1)
    
    # Load and process data
    df = load_and_process_data(json_file)
    
    # Display basic information
    print("\n--- Data Summary ---")
    print(f"Total successful trials with complete metrics: {len(df)}")
    print("\nSample Data:")
    print(df.head())
    
    # Perform statistical analysis on MAE
    print("\n--- Performing ANOVA on MAE ---")
    try:
        anova_mae = perform_statistical_analysis(df, response_variable='mae')
        print(anova_mae)
    except Exception as e:
        print(f"ANOVA failed: {e}")
        return
    
    # Calculate variance of MAE for each combination of variables
    group_vars = ['stage', 'role', 'use_conv_prior', 'resolution']
    variance_df = calculate_variance(df, group_vars, response_variable='mae')
    print("\n--- Variance of MAE for Each Combination of Variables ---")
    print(variance_df.head())
    
    # Determine if variance is high (threshold can be adjusted)
    high_variance_threshold = variance_df['variance'].quantile(0.75)  # Top 25% as high variance
    high_variance_combinations = variance_df[variance_df['variance'] > high_variance_threshold]
    print("\n--- Combinations with High Variance ---")
    print(high_variance_combinations)
    
    # Suggest more trials if high variance exists
    if not high_variance_combinations.empty:
        print("\nSome combinations have high variance. Consider conducting more trials for these combinations:")
        for _, row in high_variance_combinations.iterrows():
            combo = ' | '.join([f"{var}={row[var]}" for var in group_vars])
            print(f" - {combo} (Variance: {row['variance']:.4f})")
    else:
        print("\nVariance is within acceptable limits. No additional trials needed.")
    
    # Create output directory for plots
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate boxplots for MAE across different stages
    plot_boxplots(
        df,
        x_var='stage',
        y_var='mae',
        hue_var='role',
        title='MAE Across Stages by Role',
        xlabel='Stage',
        ylabel='Mean Absolute Error (MAE)',
        filename=os.path.join(output_dir, 'mae_boxplot_stages.png')
    )
    
    # Generate boxplots for MAE across different roles
    plot_boxplots(
        df,
        x_var='role',
        y_var='mae',
        hue_var='resolution',
        title='MAE Across Roles by Resolution',
        xlabel='Role',
        ylabel='Mean Absolute Error (MAE)',
        filename=os.path.join(output_dir, 'mae_boxplot_roles.png')
    )
    
    # Generate variance plot
    # For better visualization, we concatenate group_vars into a single string
    plot_variance(
        variance_df,
        group_vars=['stage', 'role', 'use_conv_prior', 'resolution'],
        filename=os.path.join(output_dir, 'mae_variance_all_variables.png')
    )
    
    # Plot MAE across stages in chronological order
    plot_metrics_across_stages(
        df,
        stage_order=chronological_stages,
        response_variable='mae',
        filename=os.path.join(output_dir, 'mae_across_stages.png')
    )
    
    print(f"\nPlots have been saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
