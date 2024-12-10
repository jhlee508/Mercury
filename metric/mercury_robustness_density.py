import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the CSV into a DataFrame
file_path = "../data/jaehwan/deepseek-ai/deepseek-coder-1.3b-instruct_metric_score.csv"
data = pd.read_csv(file_path, header=None, names=["Pass@1", "Beyond Metric", "Our Metric", "Our Metric (+ Clustering)"])

# Create DataFrame
df = pd.DataFrame(data, columns=["Pass@1", "Beyond Metric", "Our Metric", "Our Metric (+ Clustering)"])

# Multiply by 100 if necessary (Check if data is already scaled)
if df["Beyond Metric"].max() <= 1:
    df = df * 100

# Plot histograms for metrics
plt.figure(figsize=(12, 6))
plt.hist(df["Beyond Metric"], bins=10, alpha=0.4, label="Beyond Metric", range=(df["Beyond Metric"].min(), df["Beyond Metric"].max()), color="green", density=True)
plt.hist(df["Our Metric (+ Clustering)"], bins=10, alpha=0.4, label="Our Metric (+ Clustering)", range=(df["Our Metric (+ Clustering)"].min(), df["Our Metric (+ Clustering)"].max()), color="red", density=True)

# KDE (Density Line)
x_beyond = df["Beyond Metric"].dropna()
x_clustering = df["Our Metric (+ Clustering)"].dropna()

# Define KDE range dynamically based on data
x_range_beyond = pd.Series(range(int(x_beyond.min()), int(x_beyond.max()) + 1))
x_range_clustering = pd.Series(range(int(x_clustering.min()), int(x_clustering.max()) + 1))

# Generate density line for "Beyond Metric"
kde_beyond = stats.gaussian_kde(x_beyond)
plt.plot(x_range_beyond, kde_beyond(x_range_beyond), color="green", linestyle='-', linewidth=2)

# Generate density line for "Our Metric (+ Clustering)"
kde_clustering = stats.gaussian_kde(x_clustering)
plt.plot(x_range_clustering, kde_clustering(x_range_clustering), color="red", linestyle='-', linewidth=2)

# Title, labels, and legend
# plt.title("Comparison of Metric Distributions", fontsize=20)
plt.xlabel("Metric Score", fontsize=18)
plt.ylabel("Density", fontsize=18)
plt.legend(fontsize=16)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("figure/metric_robustness_comparison_density.png")
