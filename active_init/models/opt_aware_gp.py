#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Gaussian Process Regression models based on GPyTorch models.

These models are often a good starting point and are further documented in the
tutorials.

`SingleTaskGP` is a single-task exact GP model that uses relatively strong priors on
the Kernel hyperparameters, which work best when covariates are normalized to the unit
cube and outcomes are standardized (zero mean, unit variance). By default, this model
uses a `Standardize` outcome transform, which applies this standardization. However,
it does not (yet) use an input transform by default.

`SingleTaskGP` model works in batch mode (each batch having its own hyperparameters).
When the training observations include multiple outputs, `SingleTaskGP` uses
batching to model outputs independently.

`SingleTaskGP` supports multiple outputs. However, as a single-task model,
`SingleTaskGP` should be used only when the outputs are independent and all
use the same training inputs. If outputs are independent but they have different
training inputs, use the `ModelListGP`. When modeling correlations between outputs,
use a multi-task model like `MultiTaskGP`.
"""

from __future__ import annotations
import warnings

from gpytorch.utils.warnings import GPInputWarning
from gpytorch.priors import NormalPrior

import torch
from torch import Tensor
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.constraints import Interval
from botorch.acquisition import (
    AcquisitionFunction,
    qNoisyExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.likelihoods.likelihood import Likelihood
from botorch.models import SingleTaskGP
from gpytorch.means import Mean, ZeroMean, ConstantMean
from gpytorch.module import Module
from torch import Tensor
from gpytorch.module import Module
from gpytorch.mlls import AddedLossTerm
from gpytorch.priors import UniformPrior, Prior
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples

from gpytorch.settings import cholesky_jitter

# Set global default jitter value
cholesky_jitter._set_value(1e-6, 1e-6, 1e-4)


class OptimizationAwareMarginalLogLikelihood(ExactMarginalLogLikelihood):
    def __init__(self, likelihood, model):
        super(ExactMarginalLogLikelihood, self).__init__(likelihood, model)

    def forward(self, function_dist, target, *params, **kwargs):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """

        # Temporarily suppress GPInputWarning
        mll = super().forward(function_dist, target, *params, **kwargs)

        # TODO these are the zero-one transformed variants - fix
        if self.model.optimize_locations:
            opt_X = self.model.opt_X.unsqueeze(-2)
        else:
            opt_X = self.model.opt_X_buffer.unsqueeze(-2)

        self.model.train_inputs[0]
        acq = qLogNoisyExpectedImprovement(
            model=self.model,
            X_baseline=self.model.train_inputs[0],
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])),
        )

        # acq evaluates posterior which sets the model in eval mode, changing it back afterwards
        # should ideally do a context manager here
        acqval = acq(opt_X).max()
        self.model.train()
        return mll + acqval


class OptimizationAwareSingleTaskGP(SingleTaskGP):

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        opt_bounds: Tensor,
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        covar_module: Module | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        acqf: AcquisitionFunction = qNoisyExpectedImprovement,
        raw_samples: int = 1024,
        optimize_locations: bool = False,
    ) -> None:
        if mean_module is None:
            mean_module = ConstantMean(constant_prior=NormalPrior(0, 0.1))

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        dim = train_X.shape[-1]
        self.optimize_locations = optimize_locations
        if self.optimize_locations:
            self.register_parameter(
                "raw_opt_X", torch.nn.Parameter(torch.randn(raw_samples, dim).flatten())
            )
            # TODO add some raw sampling procedure when registering (or re-registering) the parameter
            # i.e. not uniform, but RawAcquisitionSamplingPrior

            opt_bounds = opt_bounds.unsqueeze(-2).repeat(1, raw_samples, 1)
            self.register_prior(
                "opt_X_prior",
                UniformPrior(opt_bounds[0], opt_bounds[1]),
                self._opt_X_param,
                self._opt_X_closure,
            )
            self.register_constraint(
                "raw_opt_X", Interval(opt_bounds[0].flatten(), opt_bounds[1].flatten())
            )
        else:
            self.register_buffer("opt_X_buffer", torch.randn(raw_samples, dim))

    def _opt_X_param(self, m):
        return m.opt_X

    def _opt_X_closure(self, m, v):
        return m._set_opt_X(v)

    @property
    def opt_X(self):
        return self.raw_opt_X_constraint.transform(self.raw_opt_X).reshape(
            -1, self.train_inputs[0].shape[-1]
        )

    @opt_X.setter
    def opt_X(self, value):
        self._set_opt_X(value)

    def _set_opt_X(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).flatten.to(self.raw_opt_X)
        self.initialize(
            raw_opt_X=self.raw_opt_X_constraint.inverse_transform(value.flatten())
        )
