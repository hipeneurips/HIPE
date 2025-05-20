r"""
Active learning acquisition function specifically targeted towards initialization
and few/one-shot optimization. Kept in botorch_fb for now, will be open-sourced
once eventual paper is accepted at the latest.
"""

from __future__ import annotations
from math import log, exp, pi
import torch

from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from active_init.acquisition.active_learning import (
    qExpectedPredictiveInformationGain,
)
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import (
    PosteriorTransform,
)
from botorch.acquisition.acquisition import MCSamplerMixin
from botorch.sampling import MCSampler
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    is_fully_bayesian,
    t_batch_mode_transform,
)
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim import optimize_acqf
from active_init.acquisition.utils import (
    compute_cross_entropy,
    compute_marginal_entropy,
)
from torch import Tensor


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for Bayesian active learning. This includes:
BALD [Houlsby2011bald]_ and its batch version [kirsch2019batchbald]_.

References

.. [kirsch2019batchbald]
    Andreas Kirsch, Joost van Amersfoort, Yarin Gal.
    BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian
    Active Learning.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2019.

"""

import warnings

from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.objective import PosteriorTransform
from botorch.models import ModelListGP
from botorch.models.fully_bayesian import MCMC_DIM, SaasFullyBayesianSingleTaskGP
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    is_fully_bayesian,
    t_batch_mode_transform,
)
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from torch import Tensor


FULLY_BAYESIAN_ERROR_MSG = (
    "Fully Bayesian acquisition functions require a SaasFullyBayesianSingleTaskGP "
    "or of ModelList of SaasFullyBayesianSingleTaskGPs to run."
)

NEGATIVE_INFOGAIN_WARNING = (
    "Information gain is negative. This is likely due to a poor Monte Carlo "
    "estimation of the entropies, extremely high or extremely low correlation "
    "in the data."  # because both of those cases result in no information gain
)


def check_negative_info_gain(info_gain: Tensor) -> None:
    r"""Check if the (expected) information gain is negative, raise a warning if so."""
    if info_gain.lt(0).any():
        warnings.warn(NEGATIVE_INFOGAIN_WARNING, RuntimeWarning, stacklevel=2)


class FullyBayesianAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model: Model):
        """Base class for acquisition functions which require a Fully Bayesian
        model treatment.

        Args:
            model: A fully bayesian single-outcome model.
        """
        if is_fully_bayesian(model):
            super().__init__(model)

        else:
            raise RuntimeError(FULLY_BAYESIAN_ERROR_MSG)


class qBayesianActiveLearningByDisagreement(
    FullyBayesianAcquisitionFunction, MCSamplerMixin
):
    def __init__(
        self,
        model: ModelListGP | SaasFullyBayesianSingleTaskGP,
        sampler: MCSampler | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        num_samples: int = 512,
    ) -> None:
        """
        Batch implementation [kirsch2019batchbald]_ of BALD [Houlsby2011bald]_,
        which maximizes the mutual information between the next observation and the
        hyperparameters of the model. Computed by Monte Carlo integration.

        Args:
            model: A fully bayesian model (SaasFullyBayesianSingleTaskGP).
            sampler: The sampler used for drawing samples to approximate the entropy
                of the Gaussian Mixture posterior.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points

        """
        super().__init__(model=model)
        MCSamplerMixin.__init__(self, sampler=sampler)
        self._default_sample_shape = torch.Size([num_samples])
        self.set_X_pending(X_pending)
        self.posterior_transform = posterior_transform

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qBayesianActiveLearningByDisagreement on the candidate set `X`.
        A monte carlo-estimated information gain is computed over a Gaussian Mixture
        marginal posterior, and the Gaussian conditional posterior to obtain the
        qBayesianActiveLearningByDisagreement on the candidate set `X`.

        Args:
            X: `batch_shape x q x D`-dim Tensor of input points.

        Returns:
            A `batch_shape x num_models`-dim Tensor of BALD values.
        """
        posterior = self.model.posterior(
            X, observation_noise=True, posterior_transform=self.posterior_transform
        )
        if isinstance(posterior.mvn, MultitaskMultivariateNormal):
            # The default MultitaskMultivariateNormal conversion for
            # GuassianMixturePosteriors does not interleave (and models task and data)
            # covariances in the unintended order. This is a inter-task block-diagonal,
            # and not inter-data block-diagonal, which is the default for GMMPosteriors
            posterior.mvn._interleaved = True

        # draw samples from the mixture posterior.
        # samples: num_samples x batch_shape x num_models x q x num_outputs
        samples = self.get_posterior_samples(posterior=posterior,)

        # Estimate the entropy of 'num_samples' samples from 'num_models' models by
        # evaluating the log_prob on each sample on the mixture posterior
        # (which constitutes of M models). thus, order N*M^2 computations

        # Make room and move the model dim to the front, squeeze the num_outputs dim.
        # prev_samples: num_models x num_samples x batch_shape x 1 x q
        prev_samples = samples.unsqueeze(0).transpose(0, MCMC_DIM).squeeze(-1)

        # avg the probs over models in the mixture - dim (-2) will be broadcasted
        # with the num_models of the posterior --> querying all samples on all models
        # posterior.mvn takes q-dimensional input by default, which removes the q-dim
        # component_sample_probs: num_models x num_samples x batch_shape x num_models
        component_sample_probs = posterior.mvn.log_prob(prev_samples).exp()

        # average over mixture components
        mixture_sample_probs = component_sample_probs.mean(dim=-1, keepdim=True)

        # this is the average over the model and sample dim
        prev_entropy = -mixture_sample_probs.log().mean(dim=[0, 1])

        # the posterior entropy is an average entropy over gaussians, so no mixture
        post_entropy = -posterior.mvn.log_prob(samples.squeeze(-1)).mean(0)

        # The BALD acq is defined as an expectation over a fully bayesian model,
        # so thus, the mean is computed here andf not outside of the forward pass
        bald = (prev_entropy - post_entropy).mean(-1, keepdim=True)
        
        check_negative_info_gain(bald)
        return bald


class qInteractionInformationGain(AcquisitionFunction, MCSamplerMixin):
    def __init__(
        self,
        model: Model,
        mc_points: Tensor,
        posterior_transform: PosteriorTransform | None = None,
        sampler: MCSampler | None = None,
        X_pending: Tensor | None = None,
    ) -> None:
        """Batch implementation of Expected Predictive Information Gain (EPIG),
        which maximizes the mutual information between the subsequent queries and
        a test set of interest under the specified model. The set set may be
        randomly drawn, constitute of data from previous tasks, or a user-defined
        distribution to infuse prior knowledge.

        Args:
            model: A SingleTask or fully bayesian model.
            mc_points: A `batch_shape x N x d` tensor of points to use for
                MC-integrating the posterior entropy. Usually, these are qMC
                samples on the whole design space, but biased sampling directly
                allows weighted integration over a biased sample of test_points.
            posterior_transform: A PosteriorTransform.
            sampler: MCSampler used to approximate the entropy of the marginal
2            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points.
        """
        super().__init__(model)

        self._default_sample_shape = torch.Size([256])
        MCSamplerMixin.__init__(self, sampler=sampler)

        self.register_buffer("mc_points", mc_points)
        self.register_buffer("X_pending", X_pending)
        self.posterior_transform = posterior_transform

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        return self.compute_ig(X=X)

    def compute_ig(self, X: Tensor):
        posterior = self.model.posterior(
            X, observation_noise=True, posterior_transform=self.posterior_transform
        )

        # TODO swap for model.observation_noise() once it is implemented
        noise = (
            posterior.variance
            - self.model.posterior(
                X,
                observation_noise=False,
                posterior_transform=self.posterior_transform,
            ).variance
        )
        cond_Y = posterior.mean

        # reshapes X: batch_size * q * D to  batch_size * num_models * q * D if
        # model is fully bayesian to accommodate the condition_on_observations call
        # NOTE @hvarfner: this too can be changed once condition_on_observations has
        # a unified interface
        if is_fully_bayesian(self.model):
            cond_X = X.unsqueeze(-3).expand(*[cond_Y.shape[:-1] + X.shape[-1:]])
        else:
            cond_X = X

        # ModelListGP.condition_on_observations does not condition each sub-model on
        # each output which is what is intended, so we have to go into each submodel
        # and condition in these instead
        if isinstance(self.model, ModelListGP):
            # If we use a ScalarizedPosteriorTransform with ModelListGPs, we still
            # need to make sure that there are m output dimensions to condition each
            # model on the outputs  - i.e. a ModelListGP-specific workaround
            # for condition_on_observations
            # NOTE @hvarfner: this can be changed once condition_on_observations has
            # a unified interface
            cond_Y = cond_Y.expand(cond_X.shape[:-1] + (self.model.num_outputs,))
            noise = noise.expand(cond_X.shape[:-1] + (self.model.num_outputs,))
            # NOTE @hvarfner this is a hacky workaround for the same issue as above
            conditional_model = ModelListGP(
                *[
                    submodel.condition_on_observations(
                        cond_X, cond_Y[..., i : i + 1], noise=noise[..., i : i + 1]
                    )
                    for i, submodel in enumerate(self.model.models)
                ]
            )
        elif isinstance(self.model, SaasFullyBayesianMultiTaskGP):
            conditional_model = self.model.condition_on_observations(
                X=X,
                Y=cond_Y[0:1, :, :],
                noise=noise[0:1, :, :],
            )
        else:
            conditional_model = self.model.condition_on_observations(
                X=cond_X,
                Y=cond_Y,
                noise=noise,
            )
        # evaluate the posterior at the grid points
        with settings.propagate_grads(False):
            # note that uncond_var will not have the batch_shape of X in it, since
            # is is independent of the candidate queries

            cond_posterior = BatchedMultiOutputGPyTorchModel.posterior(
                conditional_model, 
                X=self.mc_points.unsqueeze(-2).unsqueeze(1), 
                observation_noise=True
            )
            # the argmax is independent of prev_entropy, but enforces non-negativity
            # summing over the number of objectives and mean over the number of samples
            marginal_test_set_entropy, conditional_test_set_entropy = compute_marginal_entropy(self, cond_posterior)
            # TODO check this
            test = cond_posterior.mvn.entropy()
            tsig = (marginal_test_set_entropy - cond_posterior.mvn.entropy()).mean(0)
            return tsig


class qTestSetCrossEntropy(AcquisitionFunction, MCSamplerMixin):
    def __init__(
        self,
        model: Model,
        mc_points: Tensor,
        posterior_transform: PosteriorTransform | None = None,
        sampler: MCSampler | None = None,
        X_pending: Tensor | None = None,
    ) -> None:
        """Batch implementation of Expected Predictive Information Gain (EPIG),
        which maximizes the mutual information between the subsequent queries and
        a test set of interest under the specified model. The set set may be
        randomly drawn, constitute of data from previous tasks, or a user-defined
        distribution to infuse prior knowledge.

        Args:
            model: A SingleTask or fully bayesian model.
            mc_points: A `batch_shape x N x d` tensor of points to use for
                MC-integrating the posterior entropy. Usually, these are qMC
                samples on the whole design space, but biased sampling directly
                allows weighted integration over a biased sample of test_points.
            posterior_transform: A PosteriorTransform.
            sampler: MCSampler used to approximate the entropy of the marginal
                posterior.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points.
        """
        super().__init__(model)
        MCSamplerMixin.__init__(self, sampler=sampler)
        if mc_points.ndim != 2:
            raise ValueError(
                f"mc_points must be a 2-dimensional tensor, but got shape "
                f"{mc_points.shape}"
            )

        self.register_buffer("mc_points", mc_points)
        self.register_buffer("X_pending", X_pending)
        self.posterior_transform = posterior_transform

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        return self.compute_ig(X=X)

    def compute_ig(self, X: Tensor, entropy_type: str = ""):
        posterior = self.model.posterior(
            X, observation_noise=True, posterior_transform=self.posterior_transform
        )

        # TODO swap for model.observation_noise() once it is implemented
        noise = (
            posterior.variance
            - self.model.posterior(
                X,
                observation_noise=False,
                posterior_transform=self.posterior_transform,
            ).variance
        )
        noise = torch.nan_to_num(noise, MIN_INFERRED_NOISE_LEVEL)
        cond_Y = posterior.mean

        # reshapes X: batch_size * q * D to  batch_size * num_models * q * D if
        # model is fully bayesian to accommodate the condition_on_observations call
        # NOTE @hvarfner: this too can be changed once condition_on_observations has
        # a unified interface
        if is_fully_bayesian(self.model):
            cond_X = X.unsqueeze(-3).expand(*[cond_Y.shape[:-1] + X.shape[-1:]])
        else:
            cond_X = X

        # ModelListGP.condition_on_observations does not condition each sub-model on
        # each output which is what is intended, so we have to go into each submodel
        # and condition in these instead
        if isinstance(self.model, ModelListGP):
            # If we use a ScalarizedPosteriorTransform with ModelListGPs, we still
            # need to make sure that there are m output dimensions to condition each
            # model on the outputs  - i.e. a ModelListGP-specific workaround
            # for condition_on_observations
            # NOTE @hvarfner: this can be changed once condition_on_observations has
            # a unified interface
            cond_Y = cond_Y.expand(cond_X.shape[:-1] + (self.model.num_outputs,))
            noise = noise.expand(cond_X.shape[:-1] + (self.model.num_outputs,))
            # NOTE @hvarfner this is a hacky workaround for the same issue as above
            conditional_model = ModelListGP(
                *[
                    submodel.condition_on_observations(
                        cond_X, cond_Y[..., i : i + 1], noise=noise[..., i : i + 1]
                    )
                    for i, submodel in enumerate(self.model.models)
                ]
            )
        elif isinstance(self.model, SaasFullyBayesianMultiTaskGP):
            conditional_model = self.model.condition_on_observations(
                X=X,
                Y=cond_Y[0:1, :, :],
                noise=noise[0:1, :, :],
            )
        else:
            conditional_model = self.model.condition_on_observations(
                X=cond_X,
                Y=cond_Y,
                noise=noise,
            )
        # evaluate the posterior at the grid points
        with settings.propagate_grads(True):
            # note that uncond_var will not have the batch_shape of X in it, since
            # is is independent of the candidate queries

            cond_posterior = conditional_model.posterior(
                self.mc_points, observation_noise=True
            )
            # the argmax is independent of prev_entropy, but enforces non-negativity
            # summing over the number of objectives and mean over the number of samples
            tsce = compute_cross_entropy(self, cond_posterior)

            return -tsce


class qHyperparameterInformedPredictiveExploration(AcquisitionFunction, MCSamplerMixin):
    def __init__(
        self,
        model: Model,
        mc_points: Tensor,
        bounds: Tensor,
        posterior_transform: PosteriorTransform | None = None,
        sampler: MCSampler | None = None,
        X_pending: Tensor | None = None,
        num_samples: int = 512,
        map_model: Model | None = None,
        beta: float | int | None = None,
        q: int = 16,
        beta_tuning_method: str | float = "optimize",
    ) -> None:
        """Batch implementation of Expected Predictive Information Gain (EPIG),
        which maximizes the mutual information between the subsequent queries and
        a test set of interest under the specified model. The set set may be
        randomly drawn, constitute of data from previous tasks, or a user-defined
        distribution to infuse prior knowledge.

        Args:
            model: A SingleTask or fully bayesian model.
            mc_points: A `batch_shape x N x d` tensor of points to use for
                MC-integrating the posterior entropy. Usually, these are qMC
                samples on the whole design space, but biased sampling directly
                allows weighted integration over a biased sample of test_points.
            posterior_transform: A PosteriorTransform.
            sampler: MCSampler used to approximate the entropy of the marginal
                posterior.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points.
            beta_tuning_method: the method to optimize beta, or a float for fixed beta.
                Supports "sobol" and "optimize". 
        """
        super().__init__(model)
        MCSamplerMixin.__init__(self, sampler=sampler)
        self._default_sample_shape = torch.Size([num_samples])
        self.register_buffer("mc_points", mc_points)
        self.register_buffer("X_pending", X_pending)
        self.posterior_transform = posterior_transform
        num_models = model.posterior(model.train_inputs[0]).mean.shape[0]
        self.num_hypers = torch.cat(
            [p.reshape(num_models, -1) for p in self.model.parameters()], dim=-1
        ).shape[-1]
        self.map_model = map_model
        self.q = q
        self.tuning_method = beta_tuning_method
        self.bounds = bounds

        if beta is None:
            self._compute_tuning()

        else:
            self.tuning_factor = beta
    
    def _model_specific_condition_on_mean(self, X: Tensor) -> Model:
        """
        This method is used to condition the model on the observations.
        It is rather awkwardly implemented since the interface of 
        coniditon_on_observations is not uniform across models. 
        """
        if self.map_model is not None:
            acq_model = self.map_model
        else:
            acq_model = self.model
        posterior = acq_model.posterior(
            X, observation_noise=False, posterior_transform=self.posterior_transform
        )
 
        cond_Y = posterior.mean
        noise = (
            acq_model.posterior(
                X,
                observation_noise=True,
                posterior_transform=self.posterior_transform,
            ).variance - posterior.variance
        )
        # reshapes X: batch_size * q * D to  batch_size * num_models * q * D if
        # model is fully bayesian to accommodate the condition_on_observations call
        # NOTE @hvarfner: this too can be changed once condition_on_observations has
        # a unified interface
        if is_fully_bayesian(acq_model):
            cond_X = X.unsqueeze(-3).expand(*[cond_Y.shape[:-1] + X.shape[-1:]])
        else:
            cond_X = X
            
        # ModelListGP.condition_on_observations does not condition each sub-model on
        # each output which is what is intended, so we have to go into each submodel
        # and condition in these instead
        #with settings.propagate_grads(True):  # Huge Memory Leak
        if isinstance(acq_model, ModelListGP):
            # If we use a ScalarizedPosteriorTransform with ModelListGPs, we still
            # need to make sure that there are m output dimensions to condition each
            # model on the outputs  - i.e. a ModelListGP-specific workaround
            # for condition_on_observations
            # NOTE @hvarfner: this can be changed once condition_on_observations has
            # a unified interface
            cond_Y = cond_Y.expand(cond_X.shape[:-1] + (acq_model.num_outputs,))
            noise = noise.expand(cond_X.shape[:-1] + (acq_model.num_outputs,))
            # NOTE @hvarfner this is a hacky workaround for the same issue as above
            conditional_model = ModelListGP(
                *[
                    submodel.condition_on_observations(
                        cond_X, cond_Y[..., i : i + 1], noise=noise[..., i : i + 1]
                    )
                    for i, submodel in enumerate(acq_model.models)
                ]
            )
        elif isinstance(acq_model, SaasFullyBayesianMultiTaskGP):
            conditional_model = acq_model.condition_on_observations(
                X=X,
                Y=cond_Y[0:1, :, :],
                noise=noise[0:1, :, :],
            )
        else:
            conditional_model = acq_model.condition_on_observations(
                X=cond_X,
                Y=cond_Y,
                noise=noise,
            )
        return conditional_model, acq_model
        
    def _compute_tuning(self):
        if self.tuning_method == "sobol":
            draws = draw_sobol_samples(
                bounds=self.bounds,
                q=self.q,
                n=1,
            )
            self.tuning_factor = qConditionalHyperPredictiveInformationGain.forward(self, draws)
        
        elif self.tuning_method == "optimize":
            acqf = qConditionalHyperPredictiveInformationGain(
                model=self.model,
                mc_points=self.mc_points,
                posterior_transform=self.posterior_transform,
                sampler=self.sampler,
                X_pending=self.X_pending,
            )
            cand, self.tuning_factor = optimize_acqf(
                acqf,
                raw_samples=128,
                num_restarts=1,
                q=self.q,
                bounds=self.bounds,
                options={"batch_limit": 16}
            )
    def forward(self, X: Tensor, average: bool = True) -> Tensor:
        tsig = qExpectedPredictiveInformationGain.forward(self, X, average=average)
        if not average:
            return tsig
        bald = qBayesianActiveLearningByDisagreement.forward(self, X)
        return bald * self.tuning_factor + tsig


class qConditionalHyperPredictiveInformationGain(AcquisitionFunction, MCSamplerMixin):
    def __init__(
        self,
        model: Model,
        mc_points: Tensor,
        posterior_transform: PosteriorTransform | None = None,
        sampler: MCSampler | None = None,
        X_pending: Tensor | None = None,
    ) -> None:
        """Batch implementation of qConditionalHyperPredictiveInformationGain,
        which computes the information gain that the hyperparameters has on the test
        set given the candidate X's.

        Args:
            model: A SingleTask or fully bayesian model.
            mc_points: A `batch_shape x N x d` tensor of points to use for
                MC-integrating the posterior entropy. Usually, these are qMC
                samples on the whole design space, but biased sampling directly
                allows weighted integration over a biased sample of test_points.
            posterior_transform: A PosteriorTransform.
            sampler: MCSampler used to approximate the entropy of the marginal
                posterior.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points.
        """
        super().__init__(model)
        MCSamplerMixin.__init__(self, sampler=sampler)
        self.register_buffer("mc_points", mc_points)
        self.register_buffer("X_pending", X_pending)
        self.posterior_transform = posterior_transform

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        bs =  self.model.posterior(X[0:1]).mean.shape[-3]
        X = X.unsqueeze(-3)
        X = X.repeat(1, bs, 1, 1)
        obs = torch.zeros((*X.shape[:-1], 1), dtype=X.dtype)
        if isinstance(self.model.likelihood, FixedNoiseGaussianLikelihood):
            noise_val = self.model.likelihood.noise_covar.noise.mean()
            noise = torch.full_like(obs, noise_val)
        else:
            noise = None
        
        model = self.model.condition_on_observations(
            X=X,
            Y=obs,
            noise=noise
        )
        input_points_posterior = model.posterior(self.mc_points.unsqueeze(-3), observation_noise=True)
        marginal_tse = compute_marginal_entropy(
            self,
            posterior=input_points_posterior,
            sampler=SobolQMCNormalSampler(torch.Size([256]))
        )[0]
        conditional_tse = input_points_posterior.entropy().mean(-1, keepdim=True)
        return (marginal_tse - conditional_tse).mean(dim=0)
