r"""
Active learning acquisition function specifically targeted towards initialization
and few/one-shot optimization. Kept in botorch_fb for now, will be open-sourced
once eventual paper is accepted at the latest.
"""

from __future__ import annotations
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch.sampling.base import MCSampler

# helper to compute the monte carlo-estimated entropy of the marginal predictive posteriorin a fully Bayesian model
MCMC_DIM = -3


def compute_marginal_entropy(
    acqf: AcquisitionFunction, 
    posterior: GaussianMixturePosterior, 
    sampler: MCSampler | None = None
):
    if isinstance(posterior.mvn, MultitaskMultivariateNormal):
        # The default MultitaskMultivariateNormal conversion for
        # GuassianMixturePosteriors does not interleave (and models task and data)
        # covariances in the unintended order. This is a inter-task block-diagonal,
        # and not inter-data block-diagonal, which is the default for GMMPosteriors
        posterior.mvn._interleaved = True
    # draw samples from the mixture posterior.
    # samples: num_samples x batch_shape x num_models x q x num_outputs
    if sampler is None:
        samples = acqf.get_posterior_samples(posterior=posterior)
    else:
        samples = sampler(posterior)
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
    
    conditional_entropy = -posterior.mvn.log_prob(
        samples.squeeze(-1).mean(dim=[0])
    )
    return prev_entropy, conditional_entropy


def compute_cross_entropy(
    acqf: AcquisitionFunction, posterior: GaussianMixturePosterior
):
    if isinstance(posterior.mvn, MultitaskMultivariateNormal):
        # The default MultitaskMultivariateNormal conversion for
        # GuassianMixturePosteriors does not interleave (and models task and data)
        # covariances in the unintended order. This is a inter-task block-diagonal,
        # and not inter-data block-diagonal, which is the default for GMMPosteriors
        posterior.mvn._interleaved = True
    # draw samples from the mixture posterior.
    # samples: num_samples x batch_shape x num_models x q x num_outputs
    samples = acqf.get_posterior_samples(posterior=posterior)
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

    component_sample_logprobs = posterior.mvn.log_prob(prev_samples)

    # average over mixture components
    # mixture_sample_probs = component_sample_probs.mean(dim=-1, keepdim=True)

    # this is the average over the model and sample dim
    cross_entropy = -component_sample_logprobs.mean(dim=[0, 1])
    return cross_entropy


def compute_kl_divergence(
    acqf: AcquisitionFunction, posterior: GaussianMixturePosterior
):
    if isinstance(posterior.mvn, MultitaskMultivariateNormal):
        # The default MultitaskMultivariateNormal conversion for
        # GuassianMixturePosteriors does not interleave (and models task and data)
        # covariances in the unintended order. This is a inter-task block-diagonal,
        # and not inter-data block-diagonal, which is the default for GMMPosteriors
        posterior.mvn._interleaved = True
    # draw samples from the mixture posterior.
    # samples: num_samples x batch_shape x num_models x q x num_outputs
    samples = acqf.get_posterior_samples(posterior=posterior)
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

    component_sample_logprobs = posterior.mvn.log_prob(prev_samples)

    component_sample_probs = component_sample_logprobs.exp()

    # average over mixture components
    mixture_sample_logprobs = component_sample_probs.mean(dim=-1, keepdim=True).log()

    # this is the average over the model and sample dim
    kl_divergence = (mixture_sample_logprobs - component_sample_logprobs).mean(
        dim=[0, 1]
    )
    return kl_divergence
