from fire import Fire
from experiments.performance import (
    plot_lengthscale_distributions, 
    plot_lengthscale_distributions_swarm,
    plot_sobol_indices,
)


@Fire
def main(
    path: str,
    plot_after_init: bool = True,
    functions: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    seeds: str | list[str] | None = None,
    output_file: str | list[str] | None = None,
    swarm: bool = False,
    stacked: bool = False,
    metric: str = "sobol"
):
    # Reading a JSON file into a dictionary
    if metric == "sobol":
        plot_sobol_indices(
            base_path=path,
            plot_after_init=plot_after_init,
            functions=functions,
            methods=methods,
            seeds=seeds,
            output_file=output_file,
        )
    else:
        if swarm:
            plot_lengthscale_distributions_swarm(
                base_path=path,
                plot_after_init=plot_after_init,
                functions=functions,
                methods=methods,
                seeds=seeds,
                output_file=output_file,
            )
        else:
            plot_lengthscale_distributions(
                base_path=path,
                plot_after_init=plot_after_init,
                functions=functions,
                methods=methods,
                seeds=seeds,
                output_file=output_file,
                metric=metric,
                stacked=stacked,
            )

