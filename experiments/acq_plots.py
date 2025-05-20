import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from torch import Tensor
from typing import Callable
from experiments.constants import NAMES, COLORS
import matplotlib as mpl
mpl.rcParams["font.family"] = "serif"


def plot_acquisition_function_surface(acq_func: Callable[[Tensor], Tensor], bounds: Tensor, X_pending: Tensor, show_train_data: bool = True) -> None:
    """
    Plot the acquisition function surface and overlay the X_pending points.

    Args:
        acq_func: The acquisition function to visualize.
        bounds: A tensor specifying the bounds of the input space.
        X_pending: Tensor containing the locations of pending points.
    """
    # Check if the input space is 2D
    if bounds.size(1) != 2:
        raise ValueError("This plotting function only supports 2D inputs.")

    # Define the resolution of the grid
    resolution = 61

    # Generate a grid of points within the bounds
    x = torch.linspace(bounds[0, 0], bounds[1, 0], resolution).to(torch.double)
    y = torch.linspace(bounds[0, 1], bounds[1, 1], resolution).to(torch.double)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([X.ravel(), Y.ravel()], dim=-1)

    # Evaluate the acquisition function on the grid
    with torch.no_grad():
        Z = torch.cat(
            [acq_func(chunk) for chunk in torch.split(grid.unsqueeze(-2), 256)]
        ).view(resolution, resolution).numpy()

    # Convert X_pending to numpy for plotting
    X_pending_np = X_pending.numpy()

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X.numpy(), Y.numpy(), Z, levels=50, cmap=cm.viridis, alpha=0.8)
    plt.colorbar(label="Acquisition Function Value")
    if show_train_data and (len(acq_func.model.train_inputs[0]) > 0):
        train_X = acq_func.model.train_inputs[0].numpy()
        plt.scatter(
            train_X[0, ..., 0], train_X[0, ..., 1],
            color="dodgerblue", s=100, edgecolor="black", label="train_X"
        )

    if hasattr(acq_func, "mc_points"):
        vals = acq_func(X_pending.unsqueeze(0))
        color_spectrum = False
        if color_spectrum:
            plt.scatter(
                acq_func.mc_points[..., 0, 0], acq_func.mc_points[..., 0, 1],
                c=vals.flatten().detach().argsort().argsort(), cmap="coolwarm", edgecolor="black", label="mc_points"
            )
        else:
            plt.scatter(
                acq_func.mc_points[..., 0, 0], acq_func.mc_points[..., 0, 1],
                c="pink", edgecolor="black", label="mc_points"
            )
    plt.scatter(
        X_pending_np[:, 0], X_pending_np[:, 1],
        color="dodgerblue", s=160, edgecolor="black", label="candidates"
    )
    # Add plot details
    plt.title("Acquisition Function Surface", fontsize=16)
    plt.xlabel("x1", fontsize=14)
    plt.ylabel("x2", fontsize=14)
    plt.legend(fontsize=12, loc=8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("surf.pdf")
    plt.show()


def plot_candidates(candidates: Tensor, method: str, output_path: str = None, first: bool = True) -> None:
    """
    Plot the candidates in a 2D space.

    Args:
        candidates: A tensor containing the candidate points.
    """
    if candidates.size(1) != 2:
        raise ValueError("This plotting function only supports 2D inputs.")

    if first:
        fig, ax = plt.subplots(figsize=(4.3, 4.4))
    else:
        fig, ax = plt.subplots(figsize=(3.5, 4.4))
    ax.scatter(candidates[:, 0].numpy(), candidates[:, 1].numpy(), color=COLORS.get(method, "k"), s=160, edgecolor="black", linewidths=1.75)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_title(NAMES.get(method, "noname"), fontsize=21)
    ax.set_xlabel("$x_1$", fontsize=21)
    ax.grid(True, linestyle='--', alpha=1)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not first:
        #ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=0)
    else:
        ax.set_ylabel("$x_2$", fontsize=21)
    
    fig.tight_layout(pad=0.5)  # Reduce padding
    # Optional: try using constrained_layout instead
    # fig.set_constrained_layout(True)

    # Save figure tightly cropped
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)

    plt.savefig(output_path)
    plt.show()


def plot_objective_surface(
    objective: callable,
    bounds: torch.Tensor,
    resolution: int = 100,
    title: str = "Objective Function Contour",
    xlabel: str = "$x_1$",
    ylabel: str = "$x_2$",
    cmap: str = "viridis",
    show_colorbar: bool = True
):
    """
    Plot the contour of a 2D objective function.

    Args:
        objective: The objective function to plot. Must accept a [N, 2] tensor.
        bounds: Tensor specifying input bounds, shape [2, 2] (e.g., [[0, 0], [1, 1]]).
        resolution: Number of points per axis to sample for plotting.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        cmap: Colormap for the contour plot.
        show_colorbar: Whether to show the colorbar.
    """
    # Generate grid over input space
    x1 = torch.linspace(bounds[0, 0], bounds[1, 0], resolution)
    x2 = torch.linspace(bounds[0, 1], bounds[1, 1], resolution)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")  # [resolution, resolution]

    # Flatten grid and evaluate the objective
    X_grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1)
    with torch.no_grad():
        f_vals = objective(X_grid, noise=False).numpy().reshape(resolution, resolution)

    # Plot contour
    plt.figure(figsize=(4, 4))
    contour = plt.contourf(
        X1.numpy(), X2.numpy(), f_vals,
        levels=50,  # Increase for smoother contours
        cmap=cmap
    )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    if show_colorbar:
        plt.colorbar(contour, label="Objective Value")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()