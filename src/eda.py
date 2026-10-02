"""Exploratory figures.   Usage: python -m src.eda"""
from src import config, plots
from src.data import load_dataset


def main() -> None:
    plots.set_style()
    df = load_dataset()
    plots.plot_class_distribution(df, config.FIGURES / "class_distribution.png")
    plots.plot_correlation(df, config.FIGURES / "correlation_matrix.png")
    plots.plot_feature_distributions(df, path=config.FIGURES / "feature_distributions.png")
    print(f"EDA figures written to {config.FIGURES}")


if __name__ == "__main__":
    main()
