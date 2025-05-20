import os

from copy import deepcopy
from omegaconf.dictconfig import DictConfig
import torch
from torch import Tensor


from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.model import Model
from botorch.sampling.pathwise import get_matheron_path_model
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP
from botorch.models.fully_bayesian import FullyBayesianSingleTaskGP
from botorch.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
import json
from typing import Dict

from active_init.registry.model import get_model
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import PosteriorMean


def compute_out_of_sample_best(model: SingleTaskGP, objective):
    cand_X, _ = optimize_acqf(
        PosteriorMean(model),
        raw_samples=4096,
        num_restarts=16,
        q=1,
        bounds=objective.bounds,
        options={"sample_around_best": True}
    )
    return cand_X


def compute_rmse(model: SingleTaskGP, test_X: Tensor, test_Y: Tensor) -> float:
    """
    Compute the Root Mean Square Error (RMSE) for the model on the test set.

    Args:
        model: The BoTorch model.
        test_X: Test inputs (n x d).
        test_Y: True test outputs (n x 1).

    Returns:
        rmse: The RMSE value.
    """
    model.eval()
    with torch.no_grad():
        pred_Y = model.posterior(test_X).mean.squeeze(-1)
        rmse = torch.sqrt(torch.mean((pred_Y - test_Y.flatten()) ** 2, dim=-1))
    return rmse.detach().tolist()


def compute_mll(model: SingleTaskGP, test_X: Tensor, test_Y: Tensor) -> float:
    """
    Compute the Marginal Log-Likelihood (MLL) of the model on the training set.

    Args:
        model: The BoTorch model.
        test_X: Training inputs (n x d).
        test_Y: True training outputs (n x 1).

    Returns:
        mll_value: The MLL value.
    """

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    preds = model.posterior(test_X).mvn
    with torch.no_grad():
        # TODO fully bayesian models? - logmeanexp
        mll_value = mll(preds, test_Y.flatten())  # .exp().mean().log()
    return mll_value.detach().tolist()


def get_model_hyperparameters(
    model: SingleTaskGP | SaasFullyBayesianSingleTaskGP,
) -> Dict[str, float]:
    """
    Extract the hyperparameters (lengthscales, variance, noise level) from the model.

    Args:
        model: The BoTorch model.

    Returns:
        hyperparams: A dictionary containing hyperparameter values.
    """
    if isinstance(model.covar_module, ScaleKernel):
        hyperparams = {
            "lengthscale": model.covar_module.base_kernel.lengthscale.detach().tolist(),
            "outputscale": model.covar_module.outputscale.detach().tolist(),
            "noise": model.likelihood.noise.detach().tolist(),
        }
    elif isinstance(model.covar_module, RBFKernel) or isinstance(
        model.covar_module, MaternKernel
    ):
        batch_shape = model.covar_module.lengthscale.shape[0]
        hyperparams = {
            "lengthscale": model.covar_module.lengthscale.detach().tolist(),
            "outputscale": torch.ones(batch_shape).detach().tolist(),
            "noise": model.likelihood.noise.detach().tolist(),
        }
    elif isinstance(model.covar_module, OrthogonalAdditiveKernel):
        hyperparams = {
            "lengthscale": model.covar_module.base_kernel.lengthscale.detach().tolist(),
            "coeffs_1": model.covar_module.coeffs_1.detach().tolist(),
            "coeffs_2": model.covar_module.coeffs_2.detach().tolist(),
        }
    return hyperparams


def save_results(
    rmse: float,
    mll: float,
    hyperparams: dict[str, float],
    train_X: Tensor,
    train_Y: Tensor,
    in_sample_best: Tensor,
    out_of_sample_best: Tensor,
    in_sample_best_values: Tensor,
    out_of_sample_best_values: Tensor,
    output_path: str,
    append: str,
    train_f: Tensor | None = None,
) -> None:
    """
    Save RMSE, MLL, hyperparameters, and training data to a JSON file.

    Args:
        rmse: RMSE value.
        mll: MLL value.
        hyperparams: Dictionary of hyperparameters.
        train_X: Training inputs (n x d).
        train_Y: Training outputs (n x 1).
        output_path: Path to save the JSON file.
        train_f: Optional noiseless objective function values (n x 1).
    """
    # Convert tensors to lists for JSON serialization
    train_data = {
        "train_X": train_X.tolist(),
        "train_Y": train_Y.tolist(),
        "train_f": train_f.tolist() if train_f is not None else train_Y.tolist(),
    }
    
    results = {
        "RMSE": rmse,
        "MLL": mll,
        "Hyperparameters": hyperparams,
        "TrainingData": train_data,
    }
    if in_sample_best is not None:
        results["InSampleBest"] = in_sample_best.detach().tolist()
        results["InSampleBestValues"] = in_sample_best_values.detach().tolist()
    if out_of_sample_best is not None:
        
        results["OutOfSampleBest"] = out_of_sample_best.detach().tolist()
        results["OutOfSampleBestValues"] = out_of_sample_best_values.detach().tolist()
    
    savepath = os.environ.get("SAVEPATH", "")
    full_savepath = os.path.join(savepath, output_path)
    try:    
        os.makedirs(full_savepath, exist_ok=True)
        with open(f"{full_savepath}/{append}.json", "w") as f:
            json.dump(results, f, indent=4)
    except: # Lazy way of saying "If not on the cluster, just save here"
        backup_savepath = os.path.join(savepath, "results/test")
        os.makedirs(backup_savepath, exist_ok=True)
        with open(f"{backup_savepath}/{append}.json", "w") as f:
            json.dump(results, f, indent=4)

def get_in_sample(model: Model, train_X: Tensor):
    if isinstance(model, FullyBayesianSingleTaskGP):
        new_cand = train_X[model.posterior(train_X).mixture_mean.argmax(dim=0)]
    else:
        new_cand = train_X[model.posterior(train_X).mean.argmax(dim=0)]
    return new_cand


def compute_and_save_metrics(
    objective: callable,
    cfg: DictConfig,
    train_X: Tensor,
    train_Y: Tensor,
    in_sample_X: Tensor | None = None,
    out_of_sample_X: Tensor | None = None,
    output_path: str = "results/test",
    append: str = "",
    model: Model | None = None,
    rmses: list[float] | None = None,
    mlls: list[float] | None = None,
) -> None:
    """
    Consolidated function to compute RMSE, MLL, hyperparameters, and save results.

    Args:
        objective: black-box function.
        cfg: The Hydra DictConfig.
        train_X: Training inputs (n x d).
        train_Y: Training outputs (n x 1) with noise.
        train_f: Optional noiseless objective function values (n x 1).
        output_path: Path to save results in JSON format.
        append: Name of the file appended to the output path.
    """
    # Get the model
    if not model:
        model, _ = get_model(
            objective=objective,
            train_X=train_X,
            train_Y=train_Y,
            model_kwargs=cfg.model,
            skip_kwargs=True,
        )

    # Generate Sobol test set - flag on each task to disable MLL and RMSE on the true objective
    if cfg.get("eval", True):
        tss = cfg.evaluation.test_set_size
        test_X = draw_sobol_samples(
            bounds=objective.bounds, q=1, n=tss
        ).squeeze(-2)

        # avoids negation awkwardness
        noiseless_objective = deepcopy(objective)
        noiseless_objective.objective.noise_std = 0
        noiseless_objective.noise_std = 0
        train_f = noiseless_objective(train_X).unsqueeze(-1)
        test_f = noiseless_objective(test_X).unsqueeze(-1)  
        rmse = compute_rmse(model, test_X, test_f)
        mll = compute_mll(model, test_X, test_f)

    else:
        train_f = train_Y
        rmse = None
        mll = None
        rmse_good = None
        mll_good = None
    # Compute metrics
    hyperparams = get_model_hyperparameters(model)
    if cfg.get("eval", True):
        noiseless_objective = deepcopy(objective)
        noiseless_objective.objective.noise_std = 0
        noiseless_objective.noise_std = 0
    if in_sample_X is not None:
        in_sample_f = objective(in_sample_X, noise=False)
    if out_of_sample_X is not None:
        out_of_sample_f = objective(out_of_sample_X, noise=False)
    else:
        in_sample_f = None
        out_of_sample_f = None
    # Save results
    save_results(
        rmse=rmse if rmses is None else rmses,
        mll=mll if mlls is None else mlls,
        hyperparams=hyperparams,
        train_X=train_X,
        train_Y=train_Y,
        train_f=train_f,
        in_sample_best=in_sample_X,
        in_sample_best_values=in_sample_f,
        out_of_sample_best=out_of_sample_X,
        out_of_sample_best_values=out_of_sample_f,
        output_path=output_path,
        append=append,
    )