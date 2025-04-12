import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "output"
SAVE_DIR = "analysis_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------------------
# Step 1: Define patterns and parser
# ----------------------------------------

setting_pattern = {
    "Initial points": r"Initial points\s*:\s*(\d+)",
    "Boundary points": r"Boundary points\s*:\s*(\d+)",
    "Region points": r"Region points\s*:\s*(\d+)",
    "Max epochs": r"Max epochs\s*:\s*(\d+)",
    "Max epochs (fit)": r"Max epochs \(fit\)\s*:\s*(\d+)",
    "Learning rate": r"Learning rate\s*:\s*([0-9.eE+-]+)",
    "PI weight": r"PI weight\s*:\s*([0-9.eE+-]+)",
    "Velocity": r"Velocity\s*:\s*(\d+)",
    "L1 norm": r"L1 norm\s*:\s*([0-9.eE+-]+)",
    "L2 norm": r"L2 norm\s*:\s*([0-9.eE+-]+)",
    "Max error": r"Max error\s*:\s*([0-9.eE+-]+)",
    "Execution time": r"Total execution time:\s*([0-9.eE+-]+)"
}

def find_log_paths(output_dir):
    """Collect all paths to log.txt files under output_dir."""
    paths = []
    for root, _, files in os.walk(output_dir):
        if "log.txt" in files:
            paths.append(os.path.join(root, "log.txt"))
    return paths

def parse_log_file(log_path):
    """Parse a single log.txt and return a dictionary of extracted values."""
    record = {"Path": os.path.dirname(log_path)}
    with open(log_path, "r") as f:
        content = f.read()
        for key, pattern in setting_pattern.items():
            match = re.search(pattern, content)
            if match:
                val = match.group(1)
                try:
                    val = float(val) if "." in val or "e" in val.lower() else int(val)
                except ValueError:
                    pass
                record[key] = val
            else:
                record[key] = None
    return record

# ----------------------------------------
# Step 2: Read and parse logs
# ----------------------------------------

log_paths = find_log_paths(OUTPUT_DIR)
records = [parse_log_file(p) for p in log_paths]
df = pd.DataFrame(records).sort_values("L2 norm")
df.to_csv(f"{SAVE_DIR}/pinn_log_summary.csv", index=False)

# ----------------------------------------
# Step 3: Scatter plots
# ----------------------------------------

def plot_scatter(df, metric, filename, palette="tab20"):
    """
    Generate and save a scatter plot of metric vs learning rate.

    Args:
        df (DataFrame): Data to plot.
        metric (str): Column name to use as y-axis.
        filename (str): Output file name (no extension).
        palette (str): Seaborn color palette.
    """
    n_colors = df["PI weight"].nunique()
    color_palette = sns.color_palette(palette, n_colors)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Learning rate",
        y=metric,
        hue="PI weight",
        style="Max epochs",
        palette=color_palette
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"{metric} vs Learning Rate colored by PI Weight")
    plt.xlabel("Learning Rate (log)")
    plt.ylabel(f"{metric} (log)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{filename}.pdf")
    plt.close()

for metric in ["L1 norm", "L2 norm", "Max error"]:
    fname = metric.replace(" ", "_").lower()
    plot_scatter(df, metric, fname + "_vs_lr_piweight")

# ----------------------------------------
# Step 4: Heatmaps
# ----------------------------------------

def plot_heatmap(df, metric, filename):
    """
    Generate and save a heatmap of average metric by (Learning rate, PI weight).
    """
    agg = df.groupby(["Learning rate", "PI weight"])[metric].mean().reset_index()
    pivot = agg.pivot(index="Learning rate", columns="PI weight", values=metric)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2e", cmap="YlGnBu")
    plt.title(f"Average {metric} by Hyperparameter Grid")
    plt.xlabel("PI Weight")
    plt.ylabel("Learning Rate")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/heatmap_{metric.replace(' ', '_').lower()}.pdf")
    plt.close()

for metric in ["L1 norm", "L2 norm", "Max error"]:
    plot_heatmap(df, metric, metric)

# ----------------------------------------
# Step 5: Save best configurations
# ----------------------------------------

best_df = pd.DataFrame([
    df.loc[df["L1 norm"].idxmin()],
    df.loc[df["L2 norm"].idxmin()],
    df.loc[df["Max error"].idxmin()],
], index=["Best L1", "Best L2", "Best Max Error"])

best_df.to_csv(f"{SAVE_DIR}/best_error_comparisons.csv")

print("✅ Analysis complete. Results saved to 'analysis_output/'")
