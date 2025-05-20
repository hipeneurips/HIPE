from fire import Fire
from experiments.q_ablation import plot_metric_per_q


@Fire
def main(
    path: str,
    start_q: int = 0,
    metric: str = "MLL",
    file_name: str = "init.json",
    output_file: str = None,
    show: bool = True,
    functions: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    output_path: str = None,
):
    """
    CLI for plotting metrics across batch sizes, split by benchmark function.

    Args:
        path (str): Path to the base results directory with optional prefix filtering.
        metric (str): Metric to plot ("MLL", "RMSE", "terminal_regret").
        file_name (str): Name of the JSON file to load.
        output_file (str): Path to save the generated plot.
        show (bool): Whether to display the plot.
        functions (str | list[str] | None): Optional filter for functions.
        methods (str | list[str] | None): Optional filter for methods.
    """
    plot_metric_per_q(
        base_path=path,
        metric=metric,
        file_name=file_name,
        output_file=output_file,
        show=show,
        functions=functions,
        methods=methods,
        output_path=output_path,
        start_q=start_q,
    )


if __name__ == "__main__":
    main()