# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math
from typing import Any, Dict, Optional, Tuple

import pyro
import torch
from botorch.models.fully_bayesian import (
    compute_dists,
    matern52_kernel,
    PyroModel,
    reshape_and_detach,
)
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.kernels.kernel import Kernel
from botorch.models.kernels.orthogonal_additive_kernel import (
    OrthogonalAdditiveKernel,
    _reverse_triu_indices as rti,
)
from botorch.models.fully_bayesian import SaasPyroModel
from gpytorch.likelihoods import Likelihood
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.means.mean import Mean
from torch import Tensor
from torch.nn.parameter import Parameter

r"""Prototype fully Bayesian variant of the dimension-scaled GP with LogNormal priors
for the hyperparameters. This is a prototype.
"""


def rbf_kernel(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Squared exponential kernel."""
    dist = compute_dists(X=X, lengthscale=lengthscale)
    exp_component = torch.exp(-torch.pow(dist, 2) / 2)
    return exp_component


class ScaledDimPyroModel(PyroModel):

    _ls_loc: float = -0.75
    ls_scale: float = 0.75

    @property
    def ls_loc(self) -> float:
        return self._ls_loc + math.log(self.ard_num_dims) * 0.5

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample_mean(
        self, loc: float = 0.0, scale: float = 0.25, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(loc, **tkwargs),
                torch.tensor(scale, **tkwargs),
            ),
        )

    def sample_noise(
        self, loc: float = -5.5, scale: float = 0.75, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return MIN_INFERRED_NOISE_LEVEL + pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(loc, **tkwargs),
                    torch.tensor(scale, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_lengthscale(
        self, loc: float = 0.0, scale: float = 1.0, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.ones(self.ard_num_dims).to(**tkwargs) * loc, scale
            ),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(self.ls_loc, self.ls_scale, **tkwargs)

        K = rbf_kernel(X=self.train_X, lengthscale=lengthscale)
        K = K + noise * torch.eye(self.train_X.shape[0], **tkwargs)
        if self.train_Y.shape[-2] > 0:
            pyro.sample(
                "Y",
                pyro.distributions.MultivariateNormal(
                    loc=mean.view(-1).expand(self.train_X.shape[0]),
                    covariance_matrix=K,
                ),
                obs=self.train_Y.squeeze(-1),
            )

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = RBFKernel(
            ard_num_dims=self.ard_num_dims, batch_shape=batch_shape
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"]
                * torch.ones(num_mcmc_samples, **tkwargs),
            ).clamp_min(MIN_INFERRED_NOISE_LEVEL)
        covar_module.lengthscale = reshape_and_detach(
            target=covar_module.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood


class MultitaskScaledDimPyroModel(ScaledDimPyroModel):
    r"""
    Implementation of the multi-task sparse axis-aligned subspace priors (SAAS) model.

    The multi-task model uses an ICM kernel. The data kernel is same as the single task
    SAAS model in order to handle high-dimensional parameter spaces. The task kernel
    is a Matern-5/2 kernel using learned task embeddings as the input.
    """

    def set_inputs(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor],
        task_feature: int,
        task_rank: Optional[int] = None,
    ) -> None:
        """Set the training data.

        Args:
            train_X: Training inputs (n x (d + 1))
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). If None, we infer the noise.
                Note that the inferred noise is common across all tasks.
            task_feature: The index of the task feature (`-d <= task_feature <= d`).
            task_rank: The num of learned task embeddings to be used in the task kernel.
                If omitted, use a full rank (i.e. number of tasks) kernel.
        """
        super().set_inputs(train_X, train_Y, train_Yvar)
        # obtain a list of task indicies
        all_tasks = train_X[:, task_feature].unique().to(dtype=torch.long).tolist()
        self.task_feature = task_feature
        self.num_tasks = len(all_tasks)
        self.task_rank = task_rank or self.num_tasks
        # assume there is one column for task feature
        self.ard_num_dims = self.train_X.shape[-1] - 1

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        base_idxr = torch.arange(self.ard_num_dims, **{"device": tkwargs["device"]})
        base_idxr[self.task_feature :] += 1  # exclude task feature
        task_indices = self.train_X[..., self.task_feature].to(
            device=tkwargs["device"], dtype=torch.long
        )

        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)

        lengthscale = self.sample_lengthscale(**tkwargs)
        K = rbf_kernel(X=self.train_X[..., base_idxr], lengthscale=lengthscale)

        # compute task covar matrix
        task_latent_features = self.sample_latent_features(**tkwargs)[task_indices]
        task_lengthscale = self.sample_task_lengthscale(**tkwargs)
        task_covar = rbf_kernel(X=task_latent_features, lengthscale=task_lengthscale)
        K = K.mul(task_covar)
        K = K + noise * torch.eye(self.train_X.shape[0], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[0]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_latent_features(self, **tkwargs: Any):
        return pyro.sample(
            "latent_features",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ).expand(torch.Size([self.num_tasks, self.task_rank])),
        )

    def sample_task_lengthscale(
        self, concentration: float = 6.0, rate: float = 3.0, **tkwargs: Any
    ):
        return pyro.sample(
            "task_lengthscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ).expand(torch.Size([self.task_rank])),
        )

    def load_mcmc_samples(
        self, mcmc_samples: dict[str, Tensor]
    ) -> tuple[Mean, Kernel, Likelihood, Kernel, Parameter]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["mean"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module, covar_module, likelihood = super().load_mcmc_samples(
            mcmc_samples=mcmc_samples
        )

        task_covar_module = RBFKernel(
            ard_num_dims=self.task_rank,
            batch_shape=batch_shape,
        ).to(**tkwargs)
        task_covar_module.lengthscale = reshape_and_detach(
            target=task_covar_module.lengthscale,
            new_value=mcmc_samples["task_lengthscale"],
        )
        latent_features = Parameter(
            torch.rand(
                batch_shape + torch.Size([self.num_tasks, self.task_rank]),
                requires_grad=True,
                **tkwargs,
            )
        )
        latent_features = reshape_and_detach(
            target=latent_features,
            new_value=mcmc_samples["latent_features"],
        )
        return mean_module, covar_module, likelihood, task_covar_module, latent_features


class OrthogonalAdditivePyroModel(PyroModel):

    _ls_loc: float = -1.0
    ls_scale: float = 0.75

    def __init__(self, second_order: bool = True):
        self.second_order = second_order

    @property
    def ls_loc(self) -> float:
        return self._ls_loc + math.log(1 + self.second_order) * 0.5

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.dim = train_X.shape[-1]
        self.kern = OrthogonalAdditiveKernel(
            base_kernel=RBFKernel(ard_num_dims=self.dim),
            dim=self.dim,
            second_order=self.second_order,
        )

    def sample_offset(
        self, loc: float = -2, scale: float = 1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "offset",
            pyro.distributions.LogNormal(
                torch.tensor(loc, **tkwargs),
                torch.tensor(scale, **tkwargs),
            ),
        )

    def sample_noise(
        self, loc: float = -4.0, scale: float = 1.0, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return MIN_INFERRED_NOISE_LEVEL + pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(loc, **tkwargs),
                    torch.tensor(scale, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_lengthscale(
        self, loc: float = 0.0, scale: float = 1.0, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.ones(self.dim).to(**tkwargs) * loc, scale
            ),
        )
        return lengthscale

    def sample_first_order_coeffs(
        self, loc: float = 0, scale: float = 0.5, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the component signal varianes."""
        samples = torch.ones(self.dim).to(**tkwargs)

        signals = pyro.sample(
            "coeffs_1",
            pyro.distributions.HalfCauchy(
                samples * scale,
            ),
        )
        return signals

    def sample_second_order_coeffs(
        self, loc: float = -0.5, scale: float = 0.25, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the component signal varianes."""
        samples = torch.ones((self.dim * (self.dim - 1)) // 2).to(**tkwargs)

        signals = pyro.sample(
            "coeffs_2",
            pyro.distributions.HalfCauchy(
                samples * scale,
            ),
        )
        return signals

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        offset = self.sample_offset(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(self.ls_loc, self.ls_scale, **tkwargs)
        fo_signals = self.sample_first_order_coeffs(**tkwargs)
        self.kern.offset = offset
        self.kern.coeffs_1 = fo_signals
        if self.second_order:
            so_signals = self.sample_second_order_coeffs(**tkwargs)
            so_signals = torch.cat((so_signals, torch.zeros(1)))[rti(self.dim)].reshape(
                self.dim, self.dim
            )
            self.kern.coeffs_2 = so_signals

        self.kern.base_kernel.lengthscale = lengthscale
        K = self.kern.k(self.train_X, self.train_X).sum(dim=0)
        K = K + noise * torch.eye(self.train_X.shape[0], **tkwargs)
        if self.train_Y.shape[-2] > 0:
            pyro.sample(
                "Y",
                pyro.distributions.MultivariateNormal(
                    loc=torch.zeros(1).view(-1).expand(self.train_X.shape[0]),
                    covariance_matrix=K,
                ),
                obs=self.train_Y.squeeze(-1),
            )

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ZeroMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = OrthogonalAdditiveKernel(
            base_kernel=RBFKernel(batch_shape=batch_shape, ard_num_dims=self.dim),
            dim=self.dim,
            second_order=self.second_order,
            batch_shape=batch_shape,
            dtype=self.train_X.dtype,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"]
                * torch.ones(num_mcmc_samples, **tkwargs),
            ).clamp_min(MIN_INFERRED_NOISE_LEVEL)

        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.coeffs_1 = reshape_and_detach(
            target=covar_module.coeffs_1,
            new_value=mcmc_samples["coeffs_1"],
        )
        if self.second_order:
            coeffs_2 = mcmc_samples["coeffs_2"]
            coeffs_2 = torch.cat((coeffs_2, torch.zeros(batch_shape + (1,))), dim=-1)[
                ..., rti(self.dim)
            ].reshape(*batch_shape, self.dim, self.dim)

            covar_module.coeffs_2 = coeffs_2.detach()
        covar_module.offset = reshape_and_detach(
            target=covar_module.offset,
            new_value=mcmc_samples["offset"],
        )
        return mean_module, covar_module, likelihood


class SaasPyroModelNoOutputscale(SaasPyroModel):

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        if self.train_Y.shape[-2] > 0:
            # Do not attempt to sample Y if the data is empty.
            # This leads to errors with empty data.
            K = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
            K = K + noise * torch.eye(self.train_X.shape[0], **tkwargs)
            pyro.sample(
                "Y",
                pyro.distributions.MultivariateNormal(
                    loc=mean.view(-1).expand(self.train_X.shape[0]),
                    covariance_matrix=K,
                ),
                obs=self.train_Y.squeeze(-1),
            )

    def load_mcmc_samples(
        self, mcmc_samples: dict[str, Tensor]
    ) -> tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["mean"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=torch.ones_like(covar_module.outputscale),
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood
