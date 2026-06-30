import pandas as pd
import matplotlib.pyplot as plt
import os

# Read sequential CSV
df = pd.read_csv("sequential_results.csv")

# Create output folder
os.makedirs("graphs", exist_ok=True)

# Plot
plt.figure(figsize=(10,6))
plt.bar(df["Task"], df["TimeMs"])

plt.title("Sequential Execution Time")
plt.xlabel("Task")
plt.ylabel("Time (ms)")
plt.xticks(rotation=20)

plt.tight_layout()

outfile = "graphs/sequential_execution_time.png"
plt.savefig(outfile)

print("Saved:", outfile)