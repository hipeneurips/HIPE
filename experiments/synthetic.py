from copy import deepcopy
import math
import torch
import numpy as np
from torch import Tensor
from botorch.utils.transforms import unnormalize

from botorch.test_functions.base import BaseTestProblem
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.models import SingleTaskGP


class Embedded(SyntheticTestFunction):

    def __init__(
        self,
        function: SyntheticTestFunction,
        dim=2,
        noise_std: float = 0.0,
        negate: bool = False,
        bounds: Tensor = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        assert (
            dim >= function.dim
        ), "The effective function dimensionality is larger than the embedding dimension."
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self._bounds[0 : function.dim] = function._bounds
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self.register_buffer(
            "i", torch.tensor(tuple(range(1, self.dim + 1)), dtype=torch.float)
        )
        self.embedded_function = function

    def evaluate_true(self, X: Tensor) -> Tensor:

        embedded_X = X[:, 0 : self.embedded_function.dim]
        return self.embedded_function.evaluate_true(embedded_X)


class ZeroOneObjective:
    def __init__(self, objective: BaseTestProblem):
        """
        Wrap an objective function to accept normalized inputs in [0, 1]^d.
        This is needed to circumvent unit cube scaling issues in
        condition_on_observations, which works as intended in a [0, 1]^d-space,
        but not necessarily in other bounding boxes.

        Args:
            objective (callable): The original objective function.
        """
        self.objective = objective
        self.bounds = torch.zeros_like(objective.bounds)
        self.bounds[1, :] = 1

        # Retain attributes of the original objective
        for attr in dir(objective):
            if not attr.startswith("__") and not hasattr(self, attr):
                try:
                    setattr(self, attr, getattr(objective, attr))
                except NotImplementedError:
                    continue

    def __call__(self, X: Tensor, noise: bool = True) -> Tensor:
        return self.objective(unnormalize(X, self.objective.bounds), noise=noise)

    def __deepcopy__(self, memo):
        """
        Custom deepcopy logic.
        """
        # Create a new wrapper instance
        copied = type(self)(
            objective=self.objective,
        )
        return copied