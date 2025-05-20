from fire import Fire
from experiments.performance import plot_rmse_and_mll


@Fire
def main(
    path: str,
    plot_after_init: bool = True,
    functions: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    seeds: str | list[str] | None = None,
    output_file: str | list[str] | None = None,
):
    # Reading a JSON file into a dictionary

    plot_rmse_and_mll(
        base_path=path,
        plot_after_init=plot_after_init,
        functions=functions,
        methods=methods,
        seeds=seeds,
        output_file=output_file,
    )
