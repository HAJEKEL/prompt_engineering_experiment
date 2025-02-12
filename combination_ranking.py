import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# Data: Accuracy for each (role, prior, resolution) combination
data = [
    ('role3', 'lab', 'high', 0.566667),
    ('role1', 'lab', 'high', 0.566667),
    ('role3', 'home', 'high', 0.533333),
    ('role2', 'home', 'high', 0.533333),
    ('role2', 'lab', 'high', 0.5),
    ('role2', 'lab', 'low', 0.5),
    ('role3', 'lab', 'low', 0.4),
    ('role1', 'home', 'high', 0.4),
    ('role1', 'lab', 'low', 0.366667),
    ('role3', 'none', 'high', 0.233333),
    ('role2', 'none', 'high', 0.2),
    ('role3', 'home', 'low', 0.166667),
    ('role1', 'home', 'low', 0.133333),
    ('role2', 'home', 'low', 0.133333),
    ('role3', 'none', 'low', 0.066667),
    ('role1', 'none', 'low', 0.0),
    ('role1', 'none', 'high', 0.0),
    ('role2', 'none', 'low', 0.0),
]


# Convert to DataFrame
df = pd.DataFrame(data, columns=["role", "prior", "resolution", "accuracy"])

# Number of trials per configuration
n_trials = 30  # Each configuration had 30 trials

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
print(f"Best Configuration: {df.iloc[0][['role', 'prior', 'resolution']].tolist()} - Accuracy: {best_accuracy}, CI: {best_CI}\n")

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
plt.show()

