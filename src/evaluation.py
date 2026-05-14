from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_EVALUATION_COLUMNS = [
    "vader_sentiment",
    "pysentimiento_sentiment",
]


def validate_evaluation_input(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    missing_columns = set(REQUIRED_EVALUATION_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def calculate_agreement_rate(df: pd.DataFrame) -> float:
    validate_evaluation_input(df)

    agreement_rate = (
        df["vader_sentiment"] == df["pysentimiento_sentiment"]
    ).mean()

    return float(agreement_rate)


def build_sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    validate_evaluation_input(df)

    vader_distribution = df["vader_sentiment"].value_counts().rename("vader")
    pysentimiento_distribution = (
        df["pysentimiento_sentiment"]
        .value_counts()
        .rename("pysentimiento")
    )

    distribution_df = pd.concat(
        [vader_distribution, pysentimiento_distribution],
        axis=1,
    ).fillna(0)

    distribution_df = distribution_df.astype(int)

    return distribution_df


def build_model_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    validate_evaluation_input(df)

    comparison_table = pd.crosstab(
        df["vader_sentiment"],
        df["pysentimiento_sentiment"],
        rownames=["vader"],
        colnames=["pysentimiento"],
    )

    return comparison_table


def save_sentiment_distribution_plot(
    distribution_df: pd.DataFrame,
    output_path: str = "outputs/figures/sentiment_distribution.png",
) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    distribution_df.plot(kind="bar")
    plt.title("Sentiment Distribution by Model")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return output_file


def save_evaluation_report(
    agreement_rate: float,
    distribution_df: pd.DataFrame,
    comparison_table: pd.DataFrame,
    output_path: str = "outputs/reports/model_comparison_report.txt",
) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("Glassdoor Sentiment Analysis - Model Comparison Report")
    report.append("=" * 60)
    report.append("")
    report.append(f"Agreement rate between VADER and pysentimiento: {agreement_rate:.4f}")
    report.append("")
    report.append("Sentiment distribution by model:")
    report.append(distribution_df.to_string())
    report.append("")
    report.append("Cross-tabulation: VADER vs pysentimiento")
    report.append(comparison_table.to_string())

    output_file.write_text("\n".join(report), encoding="utf-8")

    return output_file


def run_evaluation(
    input_path: str = "data/processed/glassdoor_sentiment_results.csv",
) -> None:
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_file)

    validate_evaluation_input(df)

    agreement_rate = calculate_agreement_rate(df)
    distribution_df = build_sentiment_distribution(df)
    comparison_table = build_model_comparison_table(df)

    report_path = save_evaluation_report(
        agreement_rate=agreement_rate,
        distribution_df=distribution_df,
        comparison_table=comparison_table,
    )

    figure_path = save_sentiment_distribution_plot(distribution_df)

    print("Evaluation completed successfully.")
    print(f"Agreement rate: {agreement_rate:.4f}")
    print(f"Report saved to: {report_path}")
    print(f"Figure saved to: {figure_path}")


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()