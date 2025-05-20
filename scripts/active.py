from fire import Fire
from experiments.performance import plot_rmse_nmll_over_time


@Fire
def main(
    path: str,
    metrics: str | list[str] = "MLL",
    functions: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    seeds: str | list[str] | None = None,
    output_file: str | list[str] | None = None,
    start_at: int = 0,
    end_at: int | None = None,
    batch_size: int = 1,
    relative_ranking: bool = False,
    position: str = "only",
):
    # Reading a JSON file into a dictionary
    plot_rmse_nmll_over_time(
        base_path=path,
        metrics=metrics,
        functions=functions,
        methods=methods,
        seeds=seeds,
        output_file=output_file,
        start_at=start_at,
        end_at=end_at,
        batch_size=batch_size,
        relative_ranking=relative_ranking,
        position=position,
        
    )
