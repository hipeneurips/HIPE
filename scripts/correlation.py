from fire import Fire
from experiments.performance import plot_accuracy_regret_correlation

@Fire
def main(
    path: str,
    metric: str = "sobol",  # Options: "sobol", "rmse", "mll"
    functions: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    seeds: str | list[str] | None = None,
    output_file: str | None = None,
    correlation_metric: str = "pearson",  # Options: "pearson" or "spearman"
    after_bo: bool = True,
):
    plot_accuracy_regret_correlation(
        base_path=path,
        accuracy_metric=metric,
        functions=functions,
        methods=methods,
        seeds=seeds,
        output_file=output_file,
        correlation_metric=correlation_metric,
        after_bo=after_bo,
    )

if __name__ == "__main__":
    main()