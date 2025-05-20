from fire import Fire
from experiments.performance import plot_average_simple_regret


@Fire
def main(
    path: str,
    functions: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    seeds: str | list[str] | None = None,
    output_file: str | list[str] | None = None,
    start_at: int = 0,
    end_at: int | None = None,
    batch_size: int = 1,
    split: bool = False,
    infer: bool = True,
    subtract_baseline: bool = False,
    in_sample: bool = False,
    relative_ranking: bool = False,
    show: bool = False,
    smaller: bool = False,
):
    # Reading a JSON file into a dictionary
    plot_average_simple_regret(
        base_path=path,
        functions=functions,
        methods=methods,
        seeds=seeds,
        output_file=output_file,
        start_at=start_at,
        end_at=end_at,
        batch_size=batch_size,
        split=split,
        infer=infer,
        in_sample=in_sample,
        subtract_baseline=subtract_baseline,
        relative_ranking=relative_ranking,
        show=show,
        smaller=smaller,
    )
