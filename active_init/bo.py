from time import time
import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.models.fully_bayesian import FullyBayesianSingleTaskGP
from botorch.test_functions.base import BaseTestProblem
from botorch.optim import optimize_acqf
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from active_init.registry.model import get_model
from experiments.evaluation import get_in_sample, compute_out_of_sample_best


def bo_loop(
    objective: BaseTestProblem,
    acq: dict,
    init_X: Tensor,
    init_Y: Tensor,
    budget: int,
    batch_size: int,
    model_kwargs: dict,
):
    """
    Perform the Bayesian Optimization loop.

    Args:
        objective: Callable that evaluates the objective function.
        init_X: Initial design points.
        init_Y: Objective function values at init_X.
        budget: Optimization budget (number of iterations).
        batch_size: Number of points to acquire at once.
        model_kwargs: Parameters for the model.
        acq_opt_kwargs: Parameters for the acquisition optimizer.

    Returns:
        results: A dictionary with the final model, acquisition values, and other details.
    """
    train_X = init_X.clone()
    train_Y = init_Y.clone()
    cand_X = torch.zeros((0, train_X.shape[-1])).to(train_X)
    while len(train_X) < budget:
        batch_size = min(batch_size, budget - len(train_X))
         # Initialize model with training data
        print(f"Evaluated {train_X.shape[0]} --- Best observed: {train_Y.max()}")
        model, _ = get_model(
            objective=objective,
            train_X=train_X,
            train_Y=train_Y,
            model_kwargs=model_kwargs,
            skip_kwargs=True,
        )
        new_cand = get_in_sample(model, train_X)
        cand_X_out_of_sample = compute_out_of_sample_best(model, objective)
        cand_X = torch.cat((cand_X, new_cand))
        acq_name = acq["name"]
        # Define the acquisition function
        acq_func = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=train_X,
            prune_baseline=True,
        )
        # Optimize the acquisition function to find the next point
        candidates, value = optimize_acqf(
            acq_function=acq_func,
            bounds=objective.bounds,
            q=batch_size,
            raw_samples=1024,
            num_restarts=8,
            options={
                "sample_around_best": True,
                "sequential": True,
                "batch_limit": 128,
            },
        )

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
    new_cand = get_in_sample(model, train_X)
    oos_cand = compute_out_of_sample_best(model, objective)

    cand_X = torch.cat((cand_X, new_cand))
    cand_X_out_of_sample = torch.cat((cand_X_out_of_sample, oos_cand))
    print(f"Done --- Best observed: {train_Y.max()}")
    return train_X, train_Y, cand_X, cand_X_out_of_sample, model
