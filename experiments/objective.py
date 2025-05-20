import torch
from torch import Tensor
from botorch.test_functions import (
    Hartmann,
    Ackley,
)
from botorch.utils.transforms import unnormalize
from experiments.synthetic import (
    Embedded, 
    ZeroOneObjective, 
)
from botorch.test_functions.sensitivity_analysis import Ishigami
from experiments.lcbench import LCBenchSurrogate

def get_objective_function(obj_kwargs: dict, seed: int = 0):
    objective_type = obj_kwargs.type
    name = obj_kwargs.name
    noise_std = obj_kwargs.noise_std
    bounds = obj_kwargs.bounds
    dim = len(bounds)
    negate = obj_kwargs.negate
    TEST_FUNCTIONS = {      
        "ishigami": (
            Ishigami,
            dict(noise_std=noise_std, negate=negate),
        ),
        "hartmann4": (
            Hartmann,
            dict(dim=4, noise_std=noise_std, negate=negate),
        ),
        "hartmann4_8": (
            Embedded,
            dict(function=Hartmann(dim=4), noise_std=noise_std, negate=negate, dim=dim),
        ),
        "hartmann6": (
            Hartmann,
            dict(dim=6, noise_std=noise_std, negate=negate),
        ),
        "hartmann6_12": (
            Embedded,
            dict(function=Hartmann(dim=6), noise_std=noise_std, negate=negate, dim=dim),
        ),
        "ackley4": (
            Embedded,
            dict(function=Ackley(dim=4), noise_std=noise_std, negate=negate, dim=dim),
        )
    }

    if objective_type == "test":
        function, func_kwargs = TEST_FUNCTIONS[name]
 
    elif objective_type == "svm":
            from experiments.svm import SVMReduced as function
            func_kwargs = dict(negate=True, noise_std=noise_std, dim=dim, bounds=bounds)
    
    elif objective_type == "lcbench":
        from experiments.lcbench import LCBenchSurrogate
        function = LCBenchSurrogate
        func_kwargs = dict(
            negate=False,
            noise_std=noise_std,
            bounds=bounds,
            dataset=name,
        )
    else:
        raise ValueError(f"objective_type {objective_type} does not exist.")
    # wrap in a [0, 1] normalizing objective to avoid the issues related to contidion_on_observations
    func = function(**func_kwargs)
    if len(obj_kwargs.bounds):
        func.bounds = torch.Tensor(obj_kwargs.bounds).T.to(torch.float64)
        objective = ZeroOneObjective(func)
    else:
        RuntimeWarning("Objective not normalized to unit cube, make sure it's in [0, 1] by default.")
        objective = func
    return objective