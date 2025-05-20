import os
import json
import numpy as np
import matplotlib.pyplot as plt
from fire import Fire

def plot_init_results(
    base_dir: str = "paper_plots/fig4",
    methods: str = None,
    mode: str = "metric",  # 'metric', 'hyperparams', 'sobol'
    metric: str = "MLL",  # For mode='metric': "RMSE", "MLL", "RMSE_good", "MLL_good"
    hyperparam: str = "lengthscale",  # For mode='hyperparams': "lengthscale", "outputscale", "noise"
    file_name: str = "init.json",
    plot_type: str = "box",  # For mode='metric': "box" or "bar"
    output_file: str = None,
    show: bool = True
):
    if methods is None:
        methods = sorted(os.listdir(base_dir))
    if isinstance(methods, str):
        methods = [methods]
    data = {}

    for method in methods:
        json_path = os.path.join(base_dir, method, file_name)
        if not os.path.isfile(json_path):
            print(f"Warning: {json_path} does not exist.")
            continue

        with open(json_path, "r") as f:
            content = json.load(f)
        data[method] = content

    if not data:
        print("No valid data found. Exiting.")
        return

    plt.figure(figsize=(8, 5))

    if mode == "metric":
        plot_metric_comparison(data, methods, metric, plot_type)

    elif mode == "hyperparams":
        plot_hyperparameters(data, methods, hyperparam)

    elif mode == "sobol":
        plot_sobol_indices(data, methods)

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'metric', 'hyperparams', or 'sobol'.")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Saved plot to {output_file}")

    if show:
        plt.show()
    else:
        plt.close()


def coverage_via_random_sampling(train_X, n_samples=1000):
    """
    Compute average distance from random points to their nearest observed point.

    Args:
        train_X (np.ndarray): Array of observed points [N, D].
        n_samples (int): Number of random points to sample in [0, 1]^D.

    Returns:
        float: Average distance from random points to nearest observed point.
    """
    from scipy.spatial import cKDTree
    D = train_X.shape[1]
    random_points = np.random.uniform(0, 1, size=(n_samples, D))

    tree = cKDTree(train_X)
    dists, _ = tree.query(random_points, k=1)  # Nearest neighbor distances

    return np.mean(dists)


def plot_metric_comparison(data, methods, metric, plot_type):
    metric_data = []
    for method in methods:
        values = data[method].get(metric, None)
        if values:
            metric_data.append(np.array(values))
        else:
            print(f"Warning: Metric '{metric}' not found for method '{method}'.")
            metric_data.append(np.array([]))

    if plot_type == "box":
        bplot = plt.boxplot(
            metric_data,
            patch_artist=True,
            labels=methods,
            showfliers=False
        )
        for patch in bplot["boxes"]:
            patch.set_facecolor("lightgray")
    elif plot_type == "bar":
        means = [np.mean(d) if len(d) > 0 else np.nan for d in metric_data]
        std_errs = [np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 0 else np.nan for d in metric_data]
        x = np.arange(len(methods))
        plt.bar(x, means, yerr=std_errs, capsize=5, color="lightgray")
        plt.xticks(x, methods)
    else:
        raise ValueError(f"Unsupported plot_type '{plot_type}'.")

    plt.xlabel("Initialization Method", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f"{metric} Comparison Across Methods", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)

def plot_hyperparameters(data, methods, hyperparam):
    if hyperparam not in ["lengthscale", "outputscale", "noise"]:
        raise ValueError(f"Invalid hyperparameter '{hyperparam}'.")

    plt.figure(figsize=(8, 5))

    for dim in range(2):  # Assuming 2D
        plt.subplot(1, 2, dim + 1)
        values_per_method = []
        for method in methods:
            try:
                raw_values = data[method]["Hyperparameters"][hyperparam]
                # Flatten and extract per dimension
                values = [v[0][dim] if hyperparam != "outputscale" else v for v in raw_values]
                values_per_method.append(np.array(values))
            except (KeyError, IndexError):
                print(f"Warning: Missing {hyperparam} for method '{method}'.")
                values_per_method.append(np.array([]))

        bplot = plt.boxplot(
            values_per_method,
            patch_artist=True,
            labels=methods,
            showfliers=False
        )
        for patch in bplot["boxes"]:
            patch.set_facecolor("lightgray")

        plt.xlabel("Method", fontsize=10)
        plt.ylabel(hyperparam, fontsize=10)
        plt.title(f"{hyperparam} (Dim {dim+1})", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

def plot_sobol_indices(data, methods):
    indices_per_method = []
    coverage_per_method = []

    for method in methods:
        sobol = np.array(data[method].get("SobolIndices", [np.nan, np.nan]))
        real = np.array(data[method].get("RealIndices", [np.nan, np.nan]))

        # Compute coverage gap metric
        try:
            train_X = np.array(data[method]["TrainingData"]["train_X"])
            coverage = coverage_via_random_sampling(train_X)
        except KeyError:
            coverage = np.nan

        indices_per_method.append((sobol, real))
        coverage_per_method.append(coverage)

    x = np.arange(len(methods))
    width = 0.2

    sobol_dim1 = [pair[0][0] for pair in indices_per_method]
    sobol_dim2 = [pair[0][1] for pair in indices_per_method]
    real_dim1 = [pair[1][0] for pair in indices_per_method]
    real_dim2 = [pair[1][1] for pair in indices_per_method]

    plt.figure(figsize=(10, 2.8))

    # Plot Sobol bars
    plt.bar(x - width, sobol_dim1, width, label="Sobol Dim 1", color="skyblue")
    plt.bar(x, sobol_dim2, width, label="Sobol Dim 2", color="lightgreen")

    # Offset the Coverage bar further to the right for clarity
    coverage_offset = 1.5 * width
    plt.bar(x + coverage_offset, np.sqrt(coverage_per_method), width, 
            label="Coverage Gap (Lower is Better)", color="orange", hatch="//")

    # Add horizontal lines for Real Indices (limited to the width of Sobol bars)
    for i, (real1, real2) in enumerate(zip(real_dim1, real_dim2)):
        plt.hlines(
            y=real1, xmin=x[i] - 1.5 * width, xmax=x[i] - 0.5 * width,
            linestyles="dashed", colors="black", label="Real Dim 1" if i == 0 else ""
        )
        plt.hlines(
            y=real2, xmin=x[i] - 0.5 * width, xmax=x[i] + 0.5 * width,
            linestyles="dashed", colors="gray", label="Real Dim 2" if i == 0 else ""
        )

    plt.xticks(x, methods)
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Sobol Index", fontsize=12)
    plt.title("Sobol Indices and Coverage Gap", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Add annotation to explicitly state that lower is better for coverage
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make space for legend and annotations

    plt.show()

if __name__ == "__main__":
    Fire(plot_init_results)