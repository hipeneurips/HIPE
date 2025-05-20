import torch
from torch import Tensor
import inspect
import numpy as np
import torch
from torch import Tensor

import subprocess
import os

import tempfile
from pathlib import Path
from platform import machine

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction

import gzip
import os
import time
from pathlib import Path

import numpy as np
import torch

from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import sys
import abc
from enum import Enum

import torch


class BenchmarkType(Enum):
    CONTINUOUS = 1
    BINARY = 2


class Benchmark(abc.ABC):

    def __init__(
        self,
        dim: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        type: BenchmarkType = BenchmarkType.CONTINUOUS,
    ):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.type = type

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class SVM(Benchmark):
    """
    The interior benchmark is just the benchmark in the lower-dimensional effective embedding.
    """

    def __init__(
        self,
    ):
        dim = 388
        super().__init__(
            dim=dim,
            lb=torch.zeros(dim).to(torch.float64),
            ub=torch.ones(dim).to(torch.float64),
        )
        self.X, self.y = self._load_data()
        np.random.seed(388)
        idxs = np.random.choice(
            np.arange(len(self.X)), min(2500, len(self.X)), replace=False
        )
        half = len(idxs) // 2
        self._X_train = self.X[idxs[:half]]
        self._X_test = self.X[idxs[half:]]
        self._y_train = self.y[idxs[:half]]
        self._y_test = self.y[idxs[half:]]

    def _load_data(self):
        then = time.time()
        data_folder = os.path.join(Path(__file__).parent.parent, "data", "svm")
        try:
            X = np.load(os.path.join(data_folder, "CT_slice_X.npy"))
            y = np.load(os.path.join(data_folder, "CT_slice_y.npy"))
        except:
            fx = gzip.GzipFile(os.path.join(data_folder, "CT_slice_X.npy.gz"), "r")
            fy = gzip.GzipFile(os.path.join(data_folder, "CT_slice_y.npy.gz"), "r")
            X = np.load(fx)
            y = np.load(fy)
            fx.close()
            fy.close()
        X = MinMaxScaler().fit_transform(X)
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).squeeze()
        now = time.time()
        return X, y

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x.numpy()
        C = 0.01 * (500 ** y[387])
        gamma = 0.1 * (30 ** y[386])
        epsilon = 0.01 * (100 ** y[385])
        length_scales = np.exp(4 * y[:385] - 2)

        svr = SVR(gamma=gamma, epsilon=epsilon, C=C, cache_size=1500, tol=0.001)
        svr.fit(self._X_train / length_scales, self._y_train)
        pred = svr.predict(self._X_test / length_scales)
        error = np.sqrt(np.mean(np.square(pred - self._y_test)))

        res = torch.tensor(error, dtype=torch.float64).unsqueeze(-1)
        
        return res


class SVMReduced(Benchmark):
    """
    The interior benchmark is just the benchmark in the lower-dimensional effective embedding.
    """
    _optimal_value = None
    _check_grad_at_opt: bool = False
    _bounds = None
    _optimizers = None

    def __init__(
        self,
        dim: int = 8,
    ):
        super().__init__(
            dim=dim,
            lb=torch.zeros(dim).to(torch.float64),
            ub=torch.ones(dim).to(torch.float64),
        )

        self.feature_dim = dim - 3
        self.dim = dim
        self._load_data(n_features=self.feature_dim)

    def _load_data(self, n_features: int):
        data_folder = os.path.join(Path(__file__).parent.parent, "data", "svm")
        try:
            X = np.load(os.path.join(data_folder, "CT_slice_X.npy"))
            y = np.load(os.path.join(data_folder, "CT_slice_y.npy"))
        except:
            fx = gzip.GzipFile(os.path.join(data_folder, "CT_slice_X.npy.gz"), "r")
            fy = gzip.GzipFile(os.path.join(data_folder, "CT_slice_y.npy.gz"), "r")
            X = np.load(fx)
            y = np.load(fy)
            fx.close()

        X -= X.min(axis=0)
        X = X[:, X.max(axis=0) > 1e-6]  # Throw away constant dimensions
        X = X / (X.max(axis=0) - X.min(axis=0))
        X = 2 * X - 1
        assert X.min() == -1 and X.max() == 1

        # Standardize targets
        y = (y - y.mean()) / y.std()

        # Only keep 10,000 data points and n_features features
        shuffled_indices = np.random.RandomState(0).permutation(X.shape[0])[
            :10_000
        ]  # Use seed 0
        X, y = X[shuffled_indices], y[shuffled_indices]

        # Use Xgboost to figure out feature importances and keep only the
        # most important features
        xgb = XGBRegressor(max_depth=8).fit(X, y)
        inds = (-xgb.feature_importances_).argsort()
        X = X[:, inds[:n_features]]

        # Train / Test split
        train_n = X.shape[0] // 2
        self._X_train, self._y_train = X[:train_n], y[:train_n]
        self._X_test, self._y_test = X[train_n:], y[train_n:]
        
    def __call__(self, X: Tensor) -> Tensor:
        x = X.numpy()
        
        lengthscales = x[: self.feature_dim]
        epsilon = 0.01 * 10 ** (2 * x[-3])  # Default = 0.1
        C = 0.01 * 10 ** (
            4 * x[-2]
        )  # Default = 1.0 # increased (opt at boundary prev.)
        gamma = (
            (1 / self.feature_dim)
            * 0.1
            * 10 ** (2 * x[-1])  # increased (opt at boundary prev.)
        )  # Default = 1.0 / self.x_dim
        model = SVR(
            C=C, epsilon=epsilon, gamma=gamma, tol=0.001, cache_size=1000, verbose=False
        )
        model.fit(self._X_train * lengthscales, self._y_train.copy())
        pred = model.predict(self._X_test * lengthscales)
        mse = ((pred - self._y_test) ** 2).mean(axis=0)
        return torch.Tensor([np.sqrt(mse)]).to(X)
    

class Mopta08(Benchmark):

    def __init__(self):
        dim = 124
        super().__init__(
            dim=dim,
            lb=torch.zeros(dim).to(torch.float64    ),
            ub=torch.ones(dim).to(torch.float64),
        )

        self.sysarch = 64 if sys.maxsize > 2**32 else 32
        self.machine = machine().lower()

        if self.machine == "armv7l":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_armhf.bin"
        elif self.machine == "x86_64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_elf64.bin"
        elif self.machine == "i386":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_elf32.bin"
        elif self.machine == "amd64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_amd64.exe"
        else:
            raise RuntimeError("Machine with this architecture is not supported")

        self._mopta_exectutable = os.path.join(
            Path(__file__).parent.parent, "data", "mopta08", self._mopta_exectutable
        )
        self.directory_file_descriptor = tempfile.TemporaryDirectory()
        self.directory_name = self.directory_file_descriptor.name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Mopta08 benchmark for one point
        :param x: one input configuration
        :return: value with soft constraints
        """
        x = x.squeeze()
        assert x.ndim == 1
        # write input to file in dir
        with open(os.path.join(self.directory_name, "input.txt"), "w+") as tmp_file:
            for _x in x:
                tmp_file.write(f"{_x.detach().cpu().numpy()}\n")
        # pass directory as working directory to process
        popen = subprocess.Popen(
            self._mopta_exectutable,
            stdout=subprocess.PIPE,
            cwd=self.directory_name,
        )
        popen.wait()
        # read and parse output file
        output = (
            open(os.path.join(self.directory_name, "output.txt"), "r")
            .read()
            .split("\n")
        )
        output = [x.strip() for x in output]
        output = torch.tensor(
            [float(x) for x in output if len(x) > 0],
            dtype=torch.float64,
        )
        value = output[0]
        constraints = output[1:]
        # see https://arxiv.org/pdf/2103.00349.pdf E.7
        return (
            value + 10 * torch.sum(torch.clip(constraints, min=0, max=None))
        ).unsqueeze(-1)


BENCHMARKS = {
    "mopta": Mopta08,
    "svm": SVM,
    "svm_20": SVMReduced,
    "svm_40": SVMReduced,
}


class BenchSuiteFunction(SyntheticTestFunction):
    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True,
        task_id: str = None,
        dim: int = 0,
    ) -> None:
        self.task_id = task_id
        fun = BENCHMARKS[task_id]
        if "dim" in inspect.signature(fun).parameters:
            self.f = fun(dim=dim)
        else:
            self.f = fun()

        self.dim = self.f.dim
        self._bounds = torch.cat(
            (self.f.lb.unsqueeze(0), self.f.ub.unsqueeze(0)), dim=0
        ).T
        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.cat([self._evaluate_true(x) for x in X])

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return self.f(X)


class ContainerBenchSuiteFunction(SyntheticTestFunction):
    def __init__(
        self,
        bounds: list,
        noise_std: float = 0,
        negate: bool = True,
        container: str = None,
        task_id: str = None,
    ) -> None:
        self.task_id = task_id
        self.dim = len(bounds)
        self._bounds = torch.Tensor(bounds)

        self.container = os.environ[f"bs_{container}".upper()]
        self.benchmark = task_id

        self.ARG_LIST = [self.container, "--benchmark_name", self.benchmark, "--x"]

        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.cat([self._evaluate_true(x) for x in X])

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x_str = [np.format_float_positional(x.item()) + "0" for x in X.flatten()]
        result = subprocess.run(
            self.ARG_LIST + x_str, capture_output=True, text=True, check=False
        )
        return Tensor([float(result.stdout)])
