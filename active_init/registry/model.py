import torch
from torch import Tensor
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.model import Model
from botorch.models.transforms import Standardize, Normalize
from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from gpytorch.means import ConstantMean, ZeroMean
from math import log
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import LogNormalPrior, NormalPrior
from botorch.models.fully_bayesian import SaasPyroModel
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from active_init.models.pyro_models import (
    ScaledDimPyroModel,
    OrthogonalAdditivePyroModel,
    SaasPyroModelNoOutputscale,
)
from copy import deepcopy
from botorch.fit import (
    fit_fully_bayesian_model_nuts,
    fit_gpytorch_mll,
)

MODEL_REGISTRY = {
    "map": SingleTaskGP,
    "fb": SaasFullyBayesianSingleTaskGP,
}

MODEL_FIT = {
    "map": fit_gpytorch_mll,
    "fb": fit_fully_bayesian_model_nuts,
}

PYRO_MODEL_REGISTRY = {
    "saas": SaasPyroModel,
    "scaled_dim": ScaledDimPyroModel,
    "oak": OrthogonalAdditivePyroModel,
}


def get_model(
    objective: callable,
    model_kwargs: dict,
    train_X: Tensor | None = None,
    train_Y: Tensor | None = None,
    skip_kwargs: bool = False,
    skip_train: bool = False,
) -> Model:
    model_type = model_kwargs.model_type
    model_class = MODEL_REGISTRY[model_type]
    fit_method = MODEL_FIT[model_type]
    # TODO remove fit_method, it's not general anyway
    map_model = None

    if skip_train:
        def fit_method(*args, **kwargs):
            pass

    bounds = objective.bounds
    dim = objective.dim
    # TODO set likelihood noise appropriately if there is no data
    if model_kwargs.fixed_noise and len(train_X) > 0:
        train_Yvar = torch.ones_like(train_Y) * objective.noise_std**2

    else:
        train_Yvar = None

    if train_X is None or len(train_X) == 0:
        train_X = torch.rand(0, dim).to(torch.float64)
        train_Y = torch.rand(0, 1).to(torch.float64)
        outcome_transform = None
    else:
        outcome_transform = Standardize(m=1)
    
    constructor_kwargs = {
        "train_X": train_X,
        "train_Y": train_Y,
        "train_Yvar": train_Yvar,
        "input_transform": None,
        "outcome_transform": outcome_transform,
    }
    if skip_kwargs:
        standard_fit_kwargs = model_kwargs.get("standard_fit_kwargs", {})

    if model_type == "fb_init" and len(train_X) < 2:
        base_class = RBFKernel
        ard_num_dims = train_X.shape[-1]
        lengthscale_prior = LogNormalPrior(
            loc=model_kwargs.pyro_model.kwargs._ls_loc + log(ard_num_dims) * 0.5, 
            scale=model_kwargs.pyro_model.kwargs.ls_scale
        )
        base_kernel = base_class(
            ard_num_dims=ard_num_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(
                2.5e-2, transform=None, initial_value=lengthscale_prior.mean
            ),
        )
        mean_module = ZeroMean()
        constructor_kwargs["covar_module"] = base_kernel
        constructor_kwargs["mean_module"] = mean_module
        model = FBWrapperSingleTaskGP(**constructor_kwargs)
        map_model = deepcopy(model)
        fit_fully_bayesian_from_prior(model, **model_kwargs.fit_kwargs)
    
    
    elif "fb" in model_type:
        pyro_model = PYRO_MODEL_REGISTRY[
            model_kwargs.pyro_model.name
        ]()
        if hasattr(model_kwargs.pyro_model, "kwargs"):

            for key, val in model_kwargs.pyro_model.kwargs.items():
                setattr(pyro_model, key, val)
        constructor_kwargs["pyro_model"] = pyro_model

        model = model_class(**constructor_kwargs)

        if skip_kwargs:
            fit_method(model, **standard_fit_kwargs)
        else:
            fit_method(model, **model_kwargs.fit_kwargs)
    
    if model_kwargs.fixed_noise and train_Yvar is None:
        # TODO
        # When we have fixed noise but no data, we use a standard GaussianLikelihood
        # because we get nan variances in the posterior otherwise.
        # However, the noise is not normalized, so objective.noise_std
        # is a signal-to-noise ratio (since the outputscale is 1 by default). 
        # objective.noise_std is unfortunatly shared with the noise of the objective,
        # so these config parameters should be separated in case we decide to run
        # high, fixed-noise problems.
        model.likelihood.noise = max(objective.noise_std ** 2, MIN_INFERRED_NOISE_LEVEL)
    return model, map_model
