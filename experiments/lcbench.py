from os.path import abspath, dirname, join
import torch
from torch import Tensor

from math import log10
from botorch.models.model import Model
from botorch.utils.sampling import draw_sobol_samples
from botorch.test_functions.synthetic import SyntheticTestFunction




import os
from typing import Any

import torch
from ax.benchmark.benchmark_metric import BenchmarkMetric

from ax.core.experiment import Experiment
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.registry import Models
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.testing.mock import skip_fit_gpytorch_mll_context_manager
from botorch.models import SingleTaskGP
from botorch.utils.datasets import SupervisedDataset
from gpytorch.kernels import MaternKernel
from gpytorch.priors import LogNormalPrior
from botorch.models.transforms import Normalize     
from ax.models.torch.botorch_modular.kernels import ScaleMaternKernel


METRIC_NAME = "Train/val_accuracy"
DATA_PATH = "data/lcbench/lcbench_{}.pt"


def get_lcbench_experiment(
    benchmark_name: str,
    observe_noise_stds: bool = False,
) -> Experiment:
    metric_name: str = "Train/val_accuracy"
    search_space: SearchSpace = SearchSpace(
        parameters=[
            RangeParameter(
                name="batch_size",
                parameter_type=ParameterType.INT,
                lower=16,
                upper=512,
                log_scale=True,
            ),
            RangeParameter(
                name="max_dropout",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0,  # Yes, really. Could make smaller if
                # we want to have it be more realistic.
                log_scale=False,
            ),
            RangeParameter(
                name="max_units",
                parameter_type=ParameterType.INT,
                lower=64,
                upper=1024,
                log_scale=True,
            ),
            RangeParameter(
                name="num_layers",
                parameter_type=ParameterType.INT,
                lower=1,
                upper=4,  # not a bug, even though it says 1-5 in the LCBench repo.
                # See https://github.com/automl/LCBench/issues/4
                log_scale=False,
            ),
            RangeParameter(
                name="learning_rate",
                parameter_type=ParameterType.FLOAT,
                lower=1e-4,
                upper=1e-1,
                log_scale=True,
            ),
            RangeParameter(
                name="momentum",
                parameter_type=ParameterType.FLOAT,
                lower=0.1,
                upper=0.99,
                log_scale=False,
            ),
            RangeParameter(
                name="weight_decay",
                parameter_type=ParameterType.FLOAT,
                lower=1e-5,
                upper=1e-1,
                log_scale=False,  # not a bug, see the LCBench repo.
            ),
        ]
    )
    opt_config: OptimizationConfig = OptimizationConfig(
        objective=Objective(
            metric=BenchmarkMetric(
                name=metric_name,
                lower_is_better=False,
                observe_noise_sd=observe_noise_stds,
            ),
        )
    )
    experiment = Experiment(search_space=search_space, optimization_config=opt_config)
    return experiment


def get_lcbench_surrogate() -> Surrogate:
    return Surrogate(
        botorch_model_class=SingleTaskGP,
        covar_module_class=ScaleMaternKernel,
        covar_module_options={
            "nu": 1.5,
            "ard_num_dims": 7,
            "outputscale_prior": LogNormalPrior(loc=0, scale=1),
        },
        input_transform_classes={},
    )


def get_surrogate_and_datasets(path: str) -> (
    tuple[TorchModelBridge, list[SupervisedDataset]]
):
    # We load the model hyperparameters from the saved state dict.
    obj = torch.load(path, weights_only=False)
    with skip_fit_gpytorch_mll_context_manager():
        mb = Models.BOTORCH_MODULAR(
            surrogate=get_lcbench_surrogate(),
            experiment=obj["experiment"],
            search_space=obj["experiment"].search_space,
            data=obj["data"],
        )
    mb.model.surrogate.model.load_state_dict(obj["state_dict"])
    return mb, mb.model.surrogate._last_datasets


class LCBenchSurrogate(SyntheticTestFunction):

    _bounds: list[tuple[float, float]] = [
        (log10(16), log10(512)),
        (0.0, 1.0),
        (log10(64), log10(1024)),
        (1.0, 4.0),
        (log10(1e-4), log10(1e-1)),
        (0.1, 0.99),
        (1e-5, 1e-1),
    ]
    dim: int = 7
    _surrogate_path: str = join(dirname(dirname(abspath(__file__))))

    def _load_surrogate_model(self) -> Model:
        model, _ = get_surrogate_and_datasets(join(self._surrogate_path, DATA_PATH.format(self.dataset)))
        return model

    def __init__(
        self,
        dataset: str,
        bounds: Tensor,
        noise_std: float = 0.0,
        negate: bool = False,
        use_inferred_noise: bool = False,
    ) -> None:
        self.dataset = dataset
        model = self._load_surrogate_model()
        if use_inferred_noise and noise_std == 0.0:
            noise_std = model.likelihood.noise_covar.noise.sqrt().item()
        elif use_inferred_noise and noise_std != 0.0:
            raise ValueError(
                "Cannot use inferred noise if noise_std is not 0.0, please pick one."
            )
        super().__init__(negate=negate, noise_std=noise_std)
        self.model = model
        self.param_names = self.model.parameters

    def evaluate_true(self, X: Tensor) -> Tensor:
        m = self.model.transforms["StandardizeY"].Ymean[METRIC_NAME]
        s = self.model.transforms["StandardizeY"].Ystd[METRIC_NAME]
        val = self.model.model.predict(X)[0] * s + m
        return val.flatten()