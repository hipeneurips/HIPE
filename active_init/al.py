from time import time
from copy import deepcopy
from omegaconf import OmegaConf, dictconfig

import torch

from torch import Tensor
from botorch.models.model import Model
from botorch.models.fully_bayesian import FullyBayesianSingleTaskGP
from botorch.test_functions.base import BaseTestProblem
from botorch.optim import optimize_acqf
from botorch.acquisition.utils import get_optimal_samples
from active_init.registry.model import get_model
from active_init.registry.acquisition import get_acquisition_function
from experiments.evaluation import compute_and_save_metrics
from experiments.evaluation import get_in_sample, compute_rmse, compute_mll
from botorch.utils.sampling import draw_sobol_samples
from experiments.synthetic import ZeroOneObjective

def get_rmse_and_mll(
    num_test_points: int, 
    model: Model, 
    objective: ZeroOneObjective
) -> tuple[float, float]:
    test_X = draw_sobol_samples(
        bounds=objective.bounds, q=1, n=num_test_points
    ).squeeze(-2)

    # avoids negation awkwardness
    noiseless_objective = deepcopy(objective)
    noiseless_objective.objective.noise_std = 0
    noiseless_objective.noise_std = 0
    test_f = noiseless_objective(test_X).unsqueeze(-1)

    rmse = compute_rmse(model, test_X, test_f)
    mll = compute_mll(model, test_X, test_f)
    return rmse, mll


def active_learning_loop(
    objective: BaseTestProblem,
    init_kwargs: OmegaConf,
    init_X: Tensor,
    init_Y: Tensor,
    budget: int,
    batch_size: int,
    model_kwargs: dict,
    acq_opt_kwargs: dict,
    num_test_points: int = 400,
):
    """
    Perform the Active Learning loop.

    Args:
        objective: Callable that evaluates the objective function.
        init_X: Initial design points.
        init_Y: Objective function values at init_X.
        budget: Optimization budget (number of iterations).
        batch_size: Number of points to acquire at once.
        model_kwargs: Parameters for the model.
        init_kwargs_dict: Parameters for the acquisition optimizer.
   
    Returns:
        results: A dictionary with the final model, acquisition values, and other details.
    """
    train_X = init_X.clone()
    train_Y = init_Y.clone()
    model, _ = get_model(
        objective=objective,
        train_X=train_X,
        train_Y=train_Y,
        model_kwargs=model_kwargs,
        skip_kwargs=True,
    )
    rmses, mlls = [], []
    rmse, mll = get_rmse_and_mll(num_test_points, model, objective)
    rmses.append(rmse)
    mlls.append(mll)
    while len(train_X) < budget:
         # Initialize model with training data
        print(f"Evaluated {train_X.shape[0]} --- Best observed: {train_Y.max()}")
        
        init_kwargs_dict = OmegaConf.to_container(init_kwargs)
        if init_kwargs_dict["name"] not in ["sobol", "random"]:

            acq_func = get_acquisition_function(
                objective=objective,
                acq_name=init_kwargs_dict["name"],
                model=model,
                acq_kwargs=init_kwargs_dict["acq_kwargs"],
                bounds=objective.bounds,
                mc_strategy=init_kwargs_dict.get("dist", None),
            )

            
            from gpytorch import settings
            with settings.detach_test_caches(False):
                print("Optimizing...")
                
                candidates, val = optimize_acqf(
                    acq_function=acq_func,
                    bounds=objective.bounds,
                    **acq_opt_kwargs,
                )
                
        elif init_kwargs_dict["name"] == "sobol":
            candidates = draw_sobol_samples(
                bounds=objective.bounds, q=acq_opt_kwargs.q, n=1
            ).squeeze(0)
        elif init_kwargs_dict["name"] == "random":
            candidates = torch.rand(
                (acq_opt_kwargs.q, objective.bounds.shape[-1])
            ).to(objective.bounds.device)

        new_X = candidates.detach()
        new_Y = objective(new_X).unsqueeze(-1)
        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])

        model, _ = get_model(
            objective=objective,
            train_X=train_X,
            train_Y=train_Y,
            model_kwargs=model_kwargs,
            skip_kwargs=True,
        )
        rmse, mll = get_rmse_and_mll(num_test_points, model, objective)
        rmses.append(rmse)
        mlls.append(mll)

    print(f"Done --- Best observed: {train_Y.max()}")
    return train_X, train_Y, rmses, mlls, model
