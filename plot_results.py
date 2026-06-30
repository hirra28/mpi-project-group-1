# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ─────────────────────────────────────────
# CONFIG — adjust path if needed
# ─────────────────────────────────────────
BASE_DIR = r"C:\Users\DELL\mpi-project-group-1"
SEQ_CSV = os.path.join(BASE_DIR, "sequential_results.csv")
MPI_CSV = os.path.join(BASE_DIR, "mpi_results.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

P = 4  # number of MPI nodes used in main comparison

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
seq = pd.read_csv(SEQ_CSV)
mpi = pd.read_csv(MPI_CSV)

# ═══════════════════════════════════════════════════════
# CHART 1 — Bar Chart: Sequential vs MPI per Task
# (uses the medium dataset size, 4 nodes)
# ═══════════════════════════════════════════════════════
def chart1_seq_vs_mpi_bar():
    medium_size = seq["DataSize"].max()  # adjust if you have multiple sizes
    seq_subset = seq[seq["DataSize"] == medium_size]
    mpi_subset = mpi[(mpi["DataSize"] == medium_size) & (mpi["Nodes"] == P)]

    merged = pd.merge(seq_subset, mpi_subset, on="Task", suffixes=("_seq", "_mpi"))

    x = np.arange(len(merged))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, merged["TimeMs_seq"], width, label="Sequential")
    ax.bar(x + width/2, merged["TimeMs_mpi"], width, label="MPI (4 nodes)")

    ax.set_xlabel("Analytical Task")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title("Execution Time: Sequential vs MPI per Task")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["Task"], rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chart1_seq_vs_mpi.png"), dpi=150)
    plt.close()
    print("Saved chart1_seq_vs_mpi.png")


# ═══════════════════════════════════════════════════════
# CHART 2 — Line Graph: Speedup vs Dataset Size
# ═══════════════════════════════════════════════════════
def chart2_speedup_vs_datasize():
    sizes = sorted(seq["DataSize"].unique())
    speedups = []

    for size in sizes:
        seq_total = seq[seq["DataSize"] == size]["TimeMs"].sum()
        mpi_total = mpi[(mpi["DataSize"] == size) & (mpi["Nodes"] == P)]["TimeMs"].sum()
        if mpi_total > 0:
            speedups.append(seq_total / mpi_total)
        else:
            speedups.append(None)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sizes, speedups, marker="o", linewidth=2, color="darkorange")
    ax.set_xlabel("Dataset Size (number of records)")
    ax.set_ylabel("Speedup (S = T_seq / T_par)")
    ax.set_title("Speedup vs Dataset Size")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chart2_speedup_vs_datasize.png"), dpi=150)
    plt.close()
    print("Saved chart2_speedup_vs_datasize.png")


# ═══════════════════════════════════════════════════════
# CHART 3 — Grouped Bar Chart: Task-level Time Breakdown
# Across 4 Nodes (per-node timing, if available)
# If you only log total task time (not per-node), this shows
# task time at different node counts (2, 3, 4) instead.
# ═══════════════════════════════════════════════════════
def chart3_task_breakdown_nodes():
    medium_size = seq["DataSize"].max()
    subset = mpi[mpi["DataSize"] == medium_size]

    tasks = subset["Task"].unique()
    node_counts = sorted(subset["Nodes"].unique())

    x = np.arange(len(tasks))
    width = 0.8 / len(node_counts)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, n in enumerate(node_counts):
        node_data = subset[subset["Nodes"] == n].set_index("Task").reindex(tasks)["TimeMs"]
        ax.bar(x + i * width, node_data, width, label=f"{n} nodes")

        ax.set_xlabel("Analytical Task")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title("Task-Level Time Breakdown (MPI, by Node Count)")
    ax.set_xticks(x + width * (len(node_counts) - 1) / 2)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.legend(title="Nodes Used")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chart3_task_breakdown_nodes.png"), dpi=150)
    plt.close()
    print("Saved chart3_task_breakdown_nodes.png")


# ═══════════════════════════════════════════════════════
# CHART 4 — Efficiency Chart: Parallel Efficiency (%) per Task
# E = (S / P) * 100,  S = T_seq / T_par
# ═══════════════════════════════════════════════════════
def chart4_efficiency_per_task():
    medium_size = seq["DataSize"].max()
    seq_subset = seq[seq["DataSize"] == medium_size]
    mpi_subset = mpi[(mpi["DataSize"] == medium_size) & (mpi["Nodes"] == P)]

    merged = pd.merge(seq_subset, mpi_subset, on="Task", suffixes=("_seq", "_mpi"))
    merged["Speedup"] = merged["TimeMs_seq"] / merged["TimeMs_mpi"]
    merged["Efficiency"] = (merged["Speedup"] / P) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(merged["Task"], merged["Efficiency"], color="seagreen")
    ax.axhline(100, color="red", linestyle="--", label="Ideal Efficiency (100%)")
    ax.set_xlabel("Analytical Task")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title(f"Parallel Efficiency per Task (P={P} nodes)")
    ax.set_xticklabels(merged["Task"], rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chart4_efficiency_per_task.png"), dpi=150)
    plt.close()
    print("Saved chart4_efficiency_per_task.png")


# ═══════════════════════════════════════════════════════
# CHART 5 — Amdahl's Law Overlay: Theoretical vs Actual Speedup
# S_max = 1 / (f + (1-f)/P)
# f = serial fraction, estimated from your measured speedup
# ═══════════════════════════════════════════════════════
def chart5_amdahls_law():
    medium_size = seq["DataSize"].max()
    seq_total = seq[seq["DataSize"] == medium_size]["TimeMs"].sum()

    node_counts = sorted(mpi["Nodes"].unique())
    actual_speedups = []

    for n in node_counts:
        mpi_total = mpi[(mpi["DataSize"] == medium_size) & (mpi["Nodes"] == n)]["TimeMs"].sum()
        actual_speedups.append(seq_total / mpi_total if mpi_total > 0 else None)

    # Estimate serial fraction f using measured speedup at max P (Amdahl's law rearranged)
    max_p = max(node_counts)
    s_measured = actual_speedups[node_counts.index(max_p)]
    f = (1/s_measured - 1/max_p) / (1 - 1/max_p) if s_measured and s_measured > 1 else 0.1
    f = max(0.0, min(f, 1.0))  # clamp between 0 and 1

    p_range = np.linspace(1, max(node_counts), 100)
    theoretical_speedup = 1 / (f + (1 - f) / p_range)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(p_range, theoretical_speedup, label=f"Amdahl's Law (f={f:.2f})", color="navy")
    ax.plot(node_counts, actual_speedups, marker="o", linestyle="--", color="crimson", label="Actual Speedup")
    ax.set_xlabel("Number of Nodes (P)")
    ax.set_ylabel("Speedup")
    ax.set_title("Amdahl's Law: Theoretical vs Actual Speedup")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chart5_amdahls_law.png"), dpi=150)
    plt.close()
    print("Saved chart5_amdahls_law.png")


# ═══════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    chart1_seq_vs_mpi_bar()
    chart2_speedup_vs_datasize()
    chart3_task_breakdown_nodes()
    chart4_efficiency_per_task()
    chart5_amdahls_law()
    print("\nAll 5 charts saved in:", OUTPUT_DIR)