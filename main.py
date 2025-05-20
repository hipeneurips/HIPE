import warnings

# Convert warnings to errors
# warnings.filterwarnings("error")

import hydra
from omegaconf import DictConfig
import math

import torch
from experiments.objective import get_objective_function
from active_init.bo import bo_loop
from active_init.al import active_learning_loop
from active_init.init import initialize
from experiments.evaluation import compute_and_save_metrics
from linear_operator.utils.warnings import NumericalWarning
import logging

warnings.filterwarnings("ignore", category=NumericalWarning)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logging.info(str(cfg))
    logging.info(cfg.experiment.save_path)
    torch.manual_seed(cfg.seed)
    batch_size = cfg.experiment.batch_size
    objective = get_objective_function(cfg.objective, seed=cfg.seed)

    init_X, init_Y, model = initialize(
        objective=objective,
        init_kwargs=cfg.init,
        model_kwargs=cfg.model,
        acq_opt_kwargs=cfg.acq_opt,
        batch_size=batch_size,
        include_center=cfg.init.get("include_center", True),
        seed=cfg.seed,
    )

    compute_and_save_metrics(
        objective=objective,
        train_X=init_X,
        train_Y=init_Y,
        cfg=cfg,
        output_path=cfg.experiment.save_path,
        append="init",
    )

    if cfg.task == "bo" and int(cfg.experiment.budget) > len(init_X):
        train_X, train_Y, cand_X, cand_X_out_of_sample, model = bo_loop(
            objective=objective,
            acq=cfg.acq,
            init_X=init_X,
            init_Y=init_Y,
            batch_size=batch_size,
            model_kwargs=cfg.model,
            budget=cfg.experiment.budget,
        )
        if cfg.init.get("include_center", True):
            # tracking inference regret including the start point
            cand_X = torch.cat((init_X[0:1], cand_X))

        compute_and_save_metrics(
            objective=objective,
            train_X=train_X,
            train_Y=train_Y,
            in_sample_X=cand_X,
            out_of_sample_X=cand_X_out_of_sample,
            cfg=cfg,
            output_path=cfg.experiment.save_path,
            append=cfg.task,
            model=model,
        )

    elif cfg.task == "al" and cfg.experiment.budget > len(init_X):
        train_X, train_Y, rmses, mlls, model = active_learning_loop(
            objective=objective,
            init_kwargs=cfg.init,
            init_X=init_X,
            init_Y=init_Y,
            batch_size=batch_size,
            model_kwargs=cfg.model,
            acq_opt_kwargs=cfg.acq_opt,
            budget=cfg.experiment.budget,
            num_test_points=cfg.evaluation.test_set_size,
        )
        compute_and_save_metrics(
            objective=objective,
            train_X=train_X,
            train_Y=train_Y,
            in_sample_X=None,
            rmses=rmses,
            mlls=mlls,
            cfg=cfg,
            output_path=cfg.experiment.save_path,
            append=cfg.task,
            model=model,
        )


if __name__ == "__main__":
    main()
