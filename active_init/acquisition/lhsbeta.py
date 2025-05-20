import torch
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.stats import beta

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import beta
from typing import Tuple


def myks_beta(
    distances: np.ndarray,
    empirical_cdf: np.ndarray,
    dim: int,
    shape1: float,
    shape2: float
) -> float:
    """
    Compute the Kolmogorov–Smirnov statistic between empirical and Beta-distributed CDF.

    Args:
        distances: Pairwise distances from the design.
        empirical_cdf: The empirical CDF values.
        dim: Dimensionality of the input space.
        shape1: Alpha parameter of the Beta distribution.
        shape2: Beta parameter of the Beta distribution.

    Returns:
        KS distance between empirical and Beta CDF.
    """
    normalized_d = np.sort(distances) / np.sqrt(dim)
    beta_cdf = beta.cdf(normalized_d, shape1, shape2)
    return np.max(np.abs(beta_cdf - empirical_cdf))


def lhs_beta_1(
    n: int,
    m: int,
    T: int = 10**5,
    shape1: float = 2.5,
    shape2: float = 4.0
) -> torch.Tensor:
    """
    Generate a distance-optimized Latin Hypercube design using a Beta-distributed KS criterion.

    Args:
        n: Number of design points.
        m: Dimension of the design space.
        T: Number of optimization iterations.
        shape1: Alpha parameter of the target Beta distribution.
        shape2: Beta parameter of the target Beta distribution.

    Returns:
        A PyTorch tensor of shape (n, m) representing the LHS design.
    """

    def random_permutation(size: int) -> np.ndarray:
        return np.argsort(np.random.rand(size))

    # Initial Latin Hypercube design
    lhs_indices = np.stack([random_permutation(n) for _ in range(m)], axis=1)
    sample = (lhs_indices + np.random.rand(n, m)) / n

    # Initial pairwise distances and KS score
    upper_indices = np.triu_indices(n, k=1)
    pairwise_distances = squareform(pdist(sample))
    empirical_cdf = np.linspace(1, len(upper_indices[0]), len(upper_indices[0])) / len(upper_indices[0])
    ks_score = myks_beta(pairwise_distances[upper_indices], empirical_cdf, dim=m, shape1=shape1, shape2=shape2)

    # Optimization loop
    for _ in range(T):
        prev_indices = lhs_indices.copy()
        prev_sample = sample.copy()

        i, j = np.random.choice(n, size=2, replace=False)
        lhs_indices[[i, j], 0] = lhs_indices[[j, i], 0]  # Swap column 0

        sample[[i, j], :] = (lhs_indices[[i, j], :] + np.random.rand(2, m)) / n
        pairwise_distances = squareform(pdist(sample))
        new_ks_score = myks_beta(pairwise_distances[upper_indices], empirical_cdf, dim=m, shape1=shape1, shape2=shape2)

        if new_ks_score < ks_score:
            ks_score = new_ks_score  # Accept
        else:
            lhs_indices = prev_indices
            sample = prev_sample  # Reject

    return torch.tensor(sample, dtype=torch.float64)