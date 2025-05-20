import math
from omegaconf import OmegaConf
import numpy as np
import time
import torch
from torch import Tensor
import logging
from botorch.test_functions.base import BaseTestProblem
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples
from active_init.registry.model import get_model
from active_init.registry.acquisition import get_acquisition_function
from active_init.acquisition.lhsbeta import lhs_beta_1
logging.basicConfig(level=logging.INFO)


def generate_start_point(
    objective: BaseTestProblem, 
    percentile: float = 0.667, 
    centering_fraction: float = 0.5,
    num_samples: int = 5000, 
) -> Tensor:
    # NOTE fixing this seed to ensure the problem is always identical across runs
    with torch.random.fork_rng():
        torch.manual_seed(0)
        samples = (1 - centering_fraction) / 2 + centering_fraction * draw_sobol_samples(
            bounds=objective.bounds, n=num_samples, q=1
        ).squeeze(-2)
        values = objective(samples, noise=False)
    sorted_samples = values.sort().indices
    quantile_index = sorted_samples[int(percentile * num_samples)]
    quantile_sample = samples[quantile_index].unsqueeze(0)
    return quantile_sample.flatten()


def initialize(
    objective: BaseTestProblem,
    init_kwargs: dict,
    model_kwargs: dict,
    acq_opt_kwargs: dict,
    train_X: Tensor | None = None,
    train_Y: Tensor | None = None,
    batch_size: int | None = None,
    include_center: bool = True,
    plot: bool = False,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """
    Perform the Bayesian Optimization loop.

    Args:
        objective: Callable that evaluates the objective function.
        init_kwargs: Parameters for the initialization.
        model_kwargs: Parameters for the model.
        acq_kwargs: Parameters for the acquisition function.
        batch_size: Optimization budget (number of iterations) for initialization.
        acq_opt_kwargs: Parameters for the acquisition optimizer.
        train_X: Initial design points if any.
        train_Y: Objective function values at train_Y.

    Returns:
        results: A dictionary with the final model, acquisition values, and other details.
    """
    # convert to dict to pop
    if train_X is None:
        if include_center:
            #NOTE Carl we add the center as an actual observation (and not just pending)
            # because FixedNoiseGaussianLikelihood cannot support zero observations and known observation noise        
            train_X = unnormalize(torch.full(torch.Size([1, objective.dim]), 0.5).to(torch.float64), objective.bounds)
            train_Y = torch.zeros(1, 1, dtype=torch.float64)
        else:
            train_X = torch.empty(0, objective.dim).to(torch.float64)
            train_Y = torch.empty(0, 1).to(torch.float64)

    X_pending = torch.empty(0, objective.dim).to(torch.float64)
        
    acq_opt_kwargs = OmegaConf.to_container(acq_opt_kwargs)
    init_kwargs_dict = OmegaConf.to_container(init_kwargs)

    q = acq_opt_kwargs.pop("q")
        
    if batch_size is None:
        num_init_rounds = 1
    else:
        num_init_rounds = math.ceil((batch_size - len(train_X)) / q)
        q = min(q, batch_size)
    if (len(train_X) > 0) and (q > 1):
        q_first = q - len(train_X)
    else:
        q_first = q
    model = None
    if init_kwargs.get("model_based", False):
        model, map_model = get_model(
            objective=objective,
            train_X=train_X,
            train_Y=train_Y,
            model_kwargs=model_kwargs,
        )
        for init_round in range(num_init_rounds):
            init_kwargs_dict["acq_kwargs"]["map_model"] = map_model
            
            if init_round == 0:
                q_curr = q_first
            else:
                q_curr = q
                
            # Define the acquisition function
            acq_func = get_acquisition_function(
                objective=objective,
                acq_name=init_kwargs_dict["name"],
                model=model,
                acq_kwargs=init_kwargs_dict["acq_kwargs"],
                bounds=objective.bounds,
                X_pending=X_pending,
                mc_strategy=init_kwargs_dict.get("dist", None),
            )

            
            from gpytorch import settings
            with settings.detach_test_caches(False):
                logging.info("Optimizing...")
                start = time.time()
                cands, val = optimize_acqf(
                    acq_function=acq_func,
                    bounds=objective.bounds,
                    q=q_curr,
                    **acq_opt_kwargs,
                )
                
            logging.info(f"Init round {init_round}... Value: {val}, Time: {time.time() - start}")
            X_pending = torch.cat((X_pending, cands))

        if plot:
            if objective.bounds.shape[-1] > 2:
                import matplotlib.pyplot as plt
                X_plot = torch.cat((train_X, X_pending)).clone().detach()
                plt.figure(figsize=(objective.bounds.shape[1] / 2, q / 2))
                plt.imshow(X_plot, vmin=0, vmax=1)
                plt.tight_layout()
                plt.savefig("imshow.png")
                plt.show()
            else:
                logging.info("plotting...")
                plot_acquisition_function_surface(acq_func=acq_func, bounds=objective.bounds, X_pending=X_pending)
        
        train_X = torch.cat((train_X, X_pending))
        
    elif init_kwargs.name == "sobol":
        train_X = torch.cat((train_X, draw_sobol_samples(bounds=objective.bounds, n=batch_size-len(train_X), q=1).squeeze(-2)))
    
    elif init_kwargs.name == "random":
        train_X = torch.cat((train_X, unnormalize(torch.rand(batch_size-len(train_X), objective.dim), objective.bounds)))
   
    elif init_kwargs.name == "beta":
        train_X = torch.cat((train_X, torch.distributions.Beta(2, 2).rsample(
            torch.Size([batch_size-len(train_X), objective.bounds.shape[1]])))
        )   
    elif init_kwargs.name == "lhsbeta":
        np.random.seed(seed)
        train_X = torch.cat((
            train_X, 
            lhs_beta_1(
                n=batch_size-len(train_X),
                m=objective.dim,
        )))
        train_X  = unnormalize(train_X, objective.bounds)
    torch.manual_seed(seed)
    train_Y = objective(train_X).unsqueeze(-1).to(objective.bounds)
    
    return train_X, train_Y, model
