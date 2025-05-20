from copy import deepcopy
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
from botorch.models import SingleTaskGP
from botorch.acquisition import AcquisitionFunction
from gpytorch.priors import NormalPrior
from botorch.models import SingleTaskGP
from botorch.optim.optimize import optimize_acqf
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim.optimize import optimize_acqf
from gpytorch.means import ZeroMean, ConstantMean

CMAP = "coolwarm"


def plot_gp_and_acquisition(
    objective: callable,
    models: SingleTaskGP | list[SingleTaskGP],
    train_X: Tensor,
    train_Y: Tensor,
    bounds: Tensor,
    resolution: int = 100,
):
    """
    Plot the GP, observed data, and the acquisition function for a 1D objective.

    Args:
        objective: The black-box function.
        model: The trained Gaussian Process model.
        train_X: Observed inputs.
        train_Y: Observed outputs.
        acq_func: The acquisition function to plot.
        bounds: Tensor specifying the bounds of the input space ([lower_bound, upper_bound]).
        resolution: Number of points to sample for plotting.
    """
    if isinstance(models, SingleTaskGP):
        models = [models]

    n_models = len(models)
    # Plot the GP
    fig, axes = plt.subplots(2, n_models, figsize=(9 * n_models, 7))
    axes = axes.reshape(2, n_models)

    for idx, model in enumerate(models):
        ax = axes[:, idx]
        model.eval()
        acq_func = qLogNoisyExpectedImprovement(model=model, X_baseline=train_X)
        # Generate test points for plotting
        X = torch.linspace(bounds[0, 0], bounds[1, 0], resolution).unsqueeze(-1)
        f = objective(X)
        # Get GP predictions
        with torch.no_grad():
            posterior = model.posterior(X)
            mean = posterior.mean.numpy().squeeze()
            lower, upper = posterior.mvn.confidence_region()
            lower = lower.numpy()
            upper = upper.numpy()
        # Get acquisition function values
        with torch.no_grad():
            acq_vals = acq_func(X.unsqueeze(-2)).exp().numpy().squeeze()

        # Plot the mean and confidence region
        ax[0].fill_between(
            X.numpy().squeeze(),
            lower,
            upper,
            color="lightblue",
            alpha=0.5,
            label="Confidence region",
        )
        ax[0].plot(X.numpy(), mean, label="GP Mean", color="blue")
        ax[0].plot(X.numpy(), f, label="Objective", color="black")

        # Plot the observed data
        ax[0].scatter(
            train_X.numpy(),
            train_Y.numpy(),
            color="red",
            label="Observations",
            zorder=10,
        )

        # Plot the acquisition function (scaled for visibility)
        ax[1].plot(
            X.numpy(),
            acq_vals,
            label="Acquisition Function",
            color="green",
            linestyle="--",
        )

        # Add labels, legend, and grid
        noise = round(model.likelihood.noise.item(), 4)
        ls = round(model.covar_module.lengthscale.item(), 4)

        ax[0].set_title(f"GP Model, $\ell = {ls}, \sigma_\\varepsilon^2 = {noise}$")
        ax[1].set_title("Acquisition Function")
        ax[0].set_xlabel("Input")
        ax[0].set_ylabel("Predictions")
        ax[1].set_ylabel("Acquisition Value")
        ax[0].grid(True, linestyle="--", alpha=0.7)
        ax[1].grid(True, linestyle="--", alpha=0.7)

        ax[0].legend()
        ax[1].legend()

    # Show plot
    plt.show()


def compute_mll_surface(
    model: SingleTaskGP,
    train_X: Tensor,
    train_Y: Tensor,
    lengthscale_range: tuple[float, float],
    noise_range: tuple[float, float],
    resolution: int = 31,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the marginal likelihood surface over lengthscale and noise.

    Args:
        train_X: Training inputs.
        train_Y: Training outputs.
        lengthscale_range: Tuple specifying the range of lengthscales (log10 scale).
        noise_range: Tuple specifying the range of noise values (log10 scale).
        resolution: Number of points to sample for each axis.

    Returns:
        lengthscales: 1D array of lengthscale values.
        noises: 1D array of noise values.
        mll_surface: 2D array of marginal log likelihood values.
    """
    lengthscales = torch.logspace(
        *torch.log10(torch.tensor(lengthscale_range)), resolution
    )
    noises = torch.logspace(*torch.log10(torch.tensor(noise_range)), resolution)
    mll_surface = torch.zeros(resolution, resolution)

    model.train()
    for i, ls in enumerate(lengthscales):
        for j, noise in enumerate(noises):
            model.covar_module.lengthscale = ls
            model.likelihood.noise = noise
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll_surface[i, j] = mll(
                model(*model.train_inputs), model.train_targets
            ).item()

    model.eval()
    return lengthscales.numpy(), noises.numpy(), mll_surface.numpy()


def compute_acq_surface(
    model: SingleTaskGP,
    train_X: Tensor,
    train_Y: Tensor,
    bounds: Tensor,
    lengthscale_range: tuple[float, float],
    noise_range: tuple[float, float],
    resolution: int = 31,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the maximum of the log expected improvement surface.

    Args:
        train_X: Training inputs.
        train_Y: Training outputs.
        bounds: Tensor specifying the bounds of the input space.
        lengthscale_range: Tuple specifying the range of lengthscales (log10 scale).
        noise_range: Tuple specifying the range of noise values (log10 scale).
        resolution: Number of points to sample for each axis.

    Returns:
        lengthscales: 1D array of lengthscale values.
        noises: 1D array of noise values.
        acq_surface: 2D array of acquisition function values.
    """
    lengthscales = torch.logspace(
        *torch.log10(torch.tensor(lengthscale_range)), resolution
    )
    noises = torch.logspace(*torch.log10(torch.tensor(noise_range)), resolution)
    acq_surface = torch.zeros(resolution, resolution)
    acq_cands = torch.zeros(resolution, resolution)

    for i, ls in enumerate(lengthscales):
        for j, noise in enumerate(noises):
            model.covar_module.lengthscale = ls
            model.likelihood.noise = noise
            acq_func = qLogNoisyExpectedImprovement(model=model, X_baseline=train_X)
            X_acq = torch.linspace(0, 1, 512).unsqueeze(-1).unsqueeze(-1)
            acq_max = acq_func(X_acq).max(dim=-1)
            cand, val = X_acq[acq_max.indices], acq_max.values
            # cand, val = optimize_acqf(
            #    acq_func,
            #    bounds=bounds,
            #    q=1,
            #    num_restarts=1,
            #    raw_samples=512,
            #    options={"max_iter": 5},
            # )
            acq_surface[i, j] = val.item()
            acq_cands[i, j] = cand.item()

    return lengthscales.numpy(), noises.numpy(), acq_surface.numpy()


def plot_surfaces(
    lengthscales: np.ndarray,
    noises: np.ndarray,
    surfaces: list[np.ndarray],
    maximizers: list[np.ndarray],
    names: list[str],
) -> None:
    """
    Plot the MLL surface, log EI surface, and their sum on log scales.

    Args:
        lengthscales: 1D array of lengthscale values.
        noises: 1D array of noise values.
        mll_surface: 2D array of marginal log likelihood values.
        acq_surface: 2D array of acquisition function values.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    X, Y = np.meshgrid(noises, lengthscales)

    for idx, [surface, maximizer, name] in enumerate(zip(surfaces, maximizers, names)):
        # MLL surface
        ax = axes[idx]
        cb = ax.contourf(X, Y, surface, cmap=CMAP, levels=100)
        fig.colorbar(cb, ax=ax)
        ax.scatter(
            maximizer[1],
            maximizer[0],
            color="limegreen",
            s=150,
            label="Maximizer",
            zorder=5,
        )
        ax.legend()
        ax.set_title(name)
        ax.set_xscale("log")  # Logarithmic X-axis
        ax.set_yscale("log")  # Logarithmic Y-axis

        ax.set_xlabel("Noise $\sigma_\\varepsilon^2$")
        ax.set_ylabel("Lengthscale $\ell$")

    plt.tight_layout()


def plot_optimization_aware_mll(
    objective: callable,
    train_X: Tensor,
    train_Y: Tensor,
    bounds: Tensor,
    resolution: int = 11,
):
    model = SingleTaskGP(
        train_X, train_Y, mean_module=ConstantMean(constant_prior=NormalPrior(0, 1))
    )
    mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
    fit_gpytorch_mll(mll)
    ref_lengthscale = model.covar_module.lengthscale.item()
    ref_noise = model.likelihood.noise.item()
    sqrt10 = 10**0.5
    lengthscale_range = ref_lengthscale * sqrt10 ** (-3), ref_lengthscale * sqrt10
    noise_range = ref_noise * sqrt10 ** (-3), ref_noise * sqrt10**3

    # Compute surfaces
    lengthscales, noises, mll_surface = compute_mll_surface(
        model=model,
        train_X=train_X,
        train_Y=train_Y,
        lengthscale_range=lengthscale_range,
        noise_range=noise_range,
        resolution=resolution,
    )
    _, _, acq_surface = compute_acq_surface(
        model=model,
        train_X=train_X,
        train_Y=train_Y,
        bounds=bounds,
        lengthscale_range=lengthscale_range,
        noise_range=noise_range,
        resolution=resolution,
    )
    sum_surface = mll_surface + acq_surface
    mll_argmax = np.argmax(mll_surface) // len(noises), np.argmax(mll_surface) % len(
        noises
    )
    acq_argmax = np.argmax(acq_surface) // len(noises), np.argmax(acq_surface) % len(
        noises
    )
    sum_argmax = np.argmax(sum_surface) // len(noises), np.argmax(sum_surface) % len(
        noises
    )

    mll_maximizer = (lengthscales[mll_argmax[0]], noises[mll_argmax[1]])
    acq_maximizer = (lengthscales[acq_argmax[0]], noises[acq_argmax[1]])
    sum_maximizer = (lengthscales[sum_argmax[0]], noises[sum_argmax[1]])

    # Plot results
    surfaces = [mll_surface, acq_surface, sum_surface]
    maximizers = [mll_maximizer, acq_maximizer, sum_maximizer]
    names = ["Marginal Log Likelihood", "Log EI", "Sum MLL and Log EI"]

    plot_surfaces(
        lengthscales=lengthscales,
        noises=noises,
        surfaces=surfaces,
        maximizers=maximizers,
        names=names,
    )
    models = [deepcopy(model) for i in range(len(maximizers))]
    for model_idx, (ls, noise) in enumerate(maximizers):
        models[model_idx].covar_module.lengthscale = ls
        models[model_idx].likelihood.noise = noise

    plot_gp_and_acquisition(
        objective=objective,
        models=models,
        train_X=train_X,
        train_Y=train_Y,
        bounds=objective.bounds,
    )

    plt.show()


def plot_query_distances_and_max(
    data: dict, 
    init_data: dict, 
    batch_size: int, 
    output_path: str | None = None,
):
    """
    Plot the distance (in X-space) of each query from the historical incumbent
    and the evolution of the incumbent value over the optimization run.

    Args:
        bo_data (dict): A dictionary containing data after the end of BO.
        init_data (dict): A dictionary containing training data after initialization.
    """
    # Extract train_X and train_f from the data
    train_X = np.array(data["TrainingData"]["train_X"])
    train_f = np.array(data["TrainingData"]["train_f"]).squeeze()

    # Initialize variables to store distances and max values
    inc_distances = []
    min_distances = []
    max_values = []
    incumbents = []
    incumbent_idx = 0
    # Iterate through queries starting from the second point
    for i in range(1, len(train_f)):
        # Find the incumbent index up to the previous query
        # Calculate the distance of the current query to the incumbent
        inc_distance = np.linalg.norm(train_X[i] - train_X[incumbent_idx])
        inc_distances.append(inc_distance)
        min_distance = np.linalg.norm(train_X[i] - train_X[:i], axis=-1).min()
        min_distances.append(min_distance)
        # Store the maximum value up to the previous query

        incumbent_idx = np.argmax(train_f[: i + 1])
        if incumbent_idx > 0:
            incumbents.append(incumbent_idx)
        max_values.append(train_f[incumbent_idx])

    # Convert distances and max_values to arrays for plotting
    inc_distances = np.array(inc_distances)
    min_distances = np.array(min_distances)
    max_values = np.array(max_values)
    incumbents = np.unique(incumbents)

    # Create the figure and axes
    titles = [
        "Distances of Queries from Historical Incumbents",
        "Distances of Queries from Closest Previous Point",
    ]
    distances = [inc_distances, min_distances]
    fig, axes = plt.subplots(len(distances) + 2, 1, figsize=(10, 10), sharex=True)

    # Plot the distances
    query_indices = np.arange(1, len(train_X))  # Start from the second query
    for idx, (distances, title) in enumerate(zip(distances, titles)):
        axes[idx].bar(query_indices, distances, color="skyblue", edgecolor="black")
        if len(incumbents) > 0:
            axes[idx].bar(
                incumbents, distances[incumbents - 1], color="orange", edgecolor="black", width=0.7
            )
            for i in range(0, len(train_X), batch_size):
                axes[idx].axvline(x=i-0.5, color="k", linewidth=3, linestyle=":")

        axes[idx].set_title(title, fontsize=16)
        axes[idx].set_ylabel("Distance in X-space", fontsize=14)
        axes[idx].grid(True, linestyle="--", alpha=0.8)

    # Plot the max value evolution
    axes[-2].plot(query_indices, max_values, color="orange", label="Max f")
    for i in range(0, len(train_X), batch_size):
        axes[-2].axvline(x=i-0.5, color="k", linewidth=3, linestyle=":")
    axes[-2].set_title("Evolution of the Maximum Objective Value", fontsize=16)
    axes[-2].set_xlabel("Query Index", fontsize=14)
    axes[-2].set_ylabel("Max Objective Value", fontsize=14)
    axes[-2].grid(True, linestyle="--", alpha=0.8)
    axes[-2].legend(fontsize=12)


    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    
    divider = make_axes_locatable(axes[-1])
    X = np.arange(train_X.shape[-1]) + 1
    ticks = np.flip(X)
    axes[-1].imshow(train_X.T)
    axes[-1].set_title("Location in search space")
    ax_sobol = divider.append_axes("left", size="50%", pad=0.05)
    ax_sobol.barh(
        X, 
        tick_label=ticks, 
        width=np.flip(init_data["SobolIndices"]), 
        color="dodgerblue", 
        edgecolor="k", 
        linewidth=0.5,
        #align="edge",
    )

    ax_sobol.grid(True, alpha=0.4)
    ax_sobol.barh(
        X, 
        tick_label=ticks,
        width=np.flip(init_data["RealIndices"]),
        alpha=0.33,
        color="orange",
        edgecolor="k",
        linewidth=0.5,
        #align="edge",
    )
    ax_sobol.set_ylim(*[0.5, X[-1] + 0.5])
    ax_sobol.set_title("Sobol index after init")
    # Adjust layout and show plot
    if output_path is not None:
        plt.savefig(output_path)
    plt.tight_layout()
    plt.show()
