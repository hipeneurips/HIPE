
from typing import Any
import torch
from copy import deepcopy
from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior, MCMC_DIM
from botorch.models import SingleTaskGP
from torch import Tensor
from torch.quasirandom import SobolEngine
from scipy.stats.qmc import LatinHypercube

import gpytorch
import pyro
from pyro.infer.mcmc import NUTS, MCMC


class FBWrapperSingleTaskGP(SingleTaskGP):

    _is_fully_bayesian = True
    _is_ensemble = True
    
    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GaussianMixturePosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `GaussianMixturePosterior` object. Includes observation noise
                if specified.
        """
        self._check_if_fitted()
        posterior = super().posterior(
            X=X.unsqueeze(MCMC_DIM),
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        posterior = GaussianMixturePosterior(distribution=posterior.distribution)
        return posterior

    def _check_if_fitted(self) -> None:
        return True
    
    def condition_on_observations(
        self, X: Tensor, Y: Tensor, **kwargs: Any
    ):
        """Conditions on additional observations for a Fully Bayesian model (either
        identical across models or unique per-model).

        Args:
            X: A `batch_shape x num_samples x d`-dim Tensor, where `d` is
                the dimension of the feature space and `batch_shape` is the number of
                sampled models.
            Y: A `batch_shape x num_samples x 1`-dim Tensor, where `d` is
                the dimension of the feature space and `batch_shape` is the number of
                sampled models.

        Returns:
            FBWrapperSingleTaskGP: A fully bayesian model conditioned on
              given observations. The returned model has `batch_shape` copies of the
              training data in case of identical observations (and `batch_shape`
              training datasets otherwise).
        """
        if X.ndim == 2 and Y.ndim == 2:
            # To avoid an error in GPyTorch when inferring the batch dimension, we add
            # the explicit batch shape here. The result is that the conditioned model
            # will have 'batch_shape' copies of the training data.
            X = X.repeat(self.batch_shape + (1, 1))
            Y = Y.repeat(self.batch_shape + (1, 1))

        elif X.ndim < Y.ndim:
            # We need to duplicate the training data to enable correct batch
            # size inference in gpytorch.
            X = X.repeat(*(Y.shape[:-2] + (1, 1)))
        return super().condition_on_observations(X, Y, **kwargs)


def fit_fully_bayesian_from_prior(
    model, 
    num_samples: int = 288, 
    warmup_steps: int = 192, 
    thinning: int = 12, 
    disable_progbar: bool = True
):

    total_prior_dim = 0
    if len(model.train_inputs[0]) <= 1:
        for _, module, prior, closure, setting_closure in model.named_priors():
            total_prior_dim += prior.sample(closure(module).shape).numel()

        dim_min = 0
        num_samples = num_samples // 2
        lhc = SobolEngine(dimension=total_prior_dim, scramble=True).draw(num_samples // thinning)
        samples_prior = {}
        for name, module, prior, closure, setting_closure in model.named_priors():
            prior_dim = closure(module).numel()
            prior_shape = closure(module).shape
            dim_max = dim_min + prior_dim
            samples = prior.icdf(lhc[..., dim_min:dim_max])
            samples_prior[name] = samples.reshape(*((num_samples // thinning,) + prior_shape))
            dim_min = dim_max

        map_model = deepcopy(model)
        model.pyro_load_from_samples(samples_prior)
        return model, map_model

    else:
        def pyro_model(x, y):
            with gpytorch.settings.fast_computations(True, True, False):
                sampled_model = model.pyro_sample_from_prior()
                output = sampled_model.likelihood(sampled_model(x))
                pyro.sample("obs", output, obs=y)
            return y

        nuts_kernel = NUTS(pyro_model)
        mcmc_run = MCMC(
            nuts_kernel,
            warmup_steps=warmup_steps,
            num_samples=num_samples,
            disable_progbar=disable_progbar
        )
        
        train_X, train_Y = model.train_inputs[0], model.train_targets
        mcmc_run.run(train_X, train_Y)
        model.pyro_load_from_samples({k: v[::thinning] for k, v in mcmc_run.get_samples().items()})