import torch
from torch import Tensor
from gpytorch.mlls import ExactMarginalLogLikelihood


def set_gp(mll: ExactMarginalLogLikelihood, lengthscale: list[float],  noise: float):
    model = mll.model

    model.covar_module.lengthscale = Tensor(lengthscale).to(
        model.covar_module.lengthscale
    )