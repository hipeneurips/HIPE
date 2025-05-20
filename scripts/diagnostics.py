import os

from fire import Fire
from experiments.plot import plot_query_distances_and_max
import json


@Fire
def main(
    run_path: str,
    seed: str, 
    batch_size: int = 16, 
    output_path: str | None = None,
):
    # Reading a JSON file into a dictionary
    f_bo = os.path.join(run_path, f"seed{seed}/bo.json")
    with open(f_bo, "r") as file:
        bo_data = json.load(file)
    f_init = os.path.join(run_path, f"seed{seed}/init.json")
    with open(f_init, "r") as file:
        init_data = json.load(file)

    plot_query_distances_and_max(
        data=bo_data, 
        init_data=init_data, 
        batch_size=batch_size,
        output_path=output_path
    )
