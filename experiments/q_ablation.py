import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re
from collections import defaultdict
from fire import Fire
from experiments.constants import (
    COLORS, 
    NAMES, 
    BENCHMARKS, 
    METRIC_NAMES,
    BO_METHOD_ORDER,
    AL_METHOD_ORDER,
)
import matplotlib as mpl


mpl.rcParams["font.family"] = "serif"

def plot_metric_per_q(
    base_path: str,
    start_q: int = 0,
    metric: str = "MLL",  # Options: "MLL", "RMSE", "terminal_regret"
    file_name: str = "init.json",
    output_file: str = None,
    show: bool = True,
    functions: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    output_path: str = None,
):
    """
    Plot a specified metric per method across batch sizes (q), split by function.

    Args:
        base_path (str): Base directory containing the experiment results.
        metric (str): Metric to plot. Options: "MLL", "RMSE", "terminal_regret".
        file_name (str): Name of the JSON file to load per seed.
        output_file (str): Optional path to save the plot.
        show (bool): Whether to display the plot.
        functions (str | list[str] | None): Optional filter for benchmark functions.
        methods (str | list[str] | None): Optional filter for methods.
    """

    bp = base_path.rstrip("/").split("/")
    base, prefix = "/".join(bp[:-1]), bp[-1]

    q_pattern = re.compile(r".*_q(\d+)$")
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # results[func][q][method]
    
    if functions:
        functions = functions.split(",")
    for folder in os.listdir(base):
        if not folder.startswith(prefix):
            continue

        match = q_pattern.match(folder)
        if not match:
            continue
        if not functions:
            functions = os.listdir(folder_path)
    
        q_value = int(match.group(1))
        if q_value < start_q:
            continue
        folder_path = os.path.join(base, folder)

        
        for func in functions:
            func_path = os.path.join(folder_path, func)
            for method in os.listdir(func_path):
                if methods:
                    if isinstance(methods, str):
                        methods = [methods]
                    if method not in methods:
                        continue

                method_path = os.path.join(func_path, method)
                seed_dirs = glob(os.path.join(method_path, "seed*/"))

                for seed_dir in seed_dirs:
                    json_path = os.path.join(seed_dir, file_name)
                    if not os.path.isfile(json_path):
                        continue

                    try:
                        with open(json_path, "r") as f:
                            data = json.load(f)

                        if metric == "MLL":
                            metric_value = -np.log(np.mean(np.exp(data["MLL"])))
                        elif metric == "MLL_good":
                            metric_value = -np.log(np.mean(np.exp(data["MLL_good"])))
                        elif metric == "RMSE":
                            metric_value = np.mean(data["RMSE"])
                        elif metric == "RMSE_good":
                            metric_value = np.mean(data["RMSE_good"])
                        elif metric == "infer":
                            metric_value = data.get("OutOfSampleBestValues", None)
                        elif metric == "train_f":
                            metric_value = max(data.get("TrainingData", None)["train_f"])
                        else:
                            raise ValueError(f"Unknown metric: {metric}")

                        if metric_value is not None:
                            results[func][q_value][method].append(metric_value)

                    except (json.JSONDecodeError, KeyError):
                        continue  # Skip invalid files

    if not results:
        print("No matching results found with the given filters.")
        return

    # Plot per function
    num_funcs = len(results)
    fig, axes = plt.subplots(1, num_funcs, figsize=(3.3 * num_funcs, 4.2), squeeze=False)
    axes = axes.flatten()

    for idx, (func, q_data) in enumerate(results.items()):
        ax = axes[idx]
        q_vals = sorted(q_data.keys())
        all_methods = sorted({m for q in q_data for m in q_data[q]})

        for method in all_methods:
            means, std_errs = [], []
            for q in q_vals:
                metrics = q_data[q][method]
                if metrics:
                    means.append(np.mean(metrics))
                    std_errs.append(np.std(metrics, ddof=1) / np.sqrt(len(metrics)))
                else:
                    means.append(np.nan)
                    std_errs.append(np.nan)

            means = np.array(means)
            std_errs = np.array(std_errs)

            label = NAMES.get(method, method)
            color = COLORS.get(method, "black")

            ax.errorbar(
                q_vals,
                means,
                yerr=std_errs,
                marker="o",
                capsize=4,
                label=label,
                color=color,
            )
        
        ax.tick_params(axis='both', which='major', labelsize=17)
        ax.set_xlabel("Batch Size", fontsize=18)
        if idx == 0:
            ax.set_ylabel(METRIC_NAMES[metric], fontsize=18)
    
        ax.set_title(BENCHMARKS[func], fontsize=22)
        ax.grid(True, linestyle="--", alpha=0.6)
        
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space for legend at bottom

    # Collect all handles and labels from the first subplot only
    handles, labels = axes[0].get_legend_handles_labels()

    # Create a single, centered legend below all subplots
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),  # Single row
        fontsize=20,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)  # Adjust vertical position as needed
    )
    if output_path:
        plt.savefig(output_path)
    
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
