from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml

from evaluation import (
    build_model_comparison_table,
    build_sentiment_distribution,
    calculate_agreement_rate,
    save_evaluation_report,
    save_sentiment_distribution_plot,
)
from exceptions import DataIngestionError, MLflowPipelineError, PreprocessingError
from ingestion import run_data_ingestion
from logger import get_logger
from preprocessing import preprocess_dataframe
from sentiment_pysentimiento import apply_pysentimiento_sentiment
from sentiment_vader import apply_vader_sentiment


logger = get_logger(__name__)


def load_params(params_path: str = "params.yaml") -> dict[str, Any]:
    path = Path(params_path)

    if not path.exists():
        raise MLflowPipelineError(f"Parameters file not found: {params_path}")

    with path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    if not isinstance(params, dict):
        raise MLflowPipelineError("params.yaml must contain a valid YAML dictionary.")

    return params


def validate_mlflow_params(params: dict[str, Any]) -> None:
    required_sections = ["data", "outputs", "mlflow", "pipeline"]

    missing_sections = set(required_sections) - set(params.keys())

    if missing_sections:
        raise MLflowPipelineError(
            f"Missing required params sections: {sorted(missing_sections)}"
        )

    required_data_keys = ["input_path", "sentiment_results_path"]
    required_output_keys = ["report_path", "figure_path"]
    required_mlflow_keys = ["tracking_uri", "experiment_name", "run_name"]

    for key in required_data_keys:
        if key not in params["data"]:
            raise MLflowPipelineError(f"Missing data parameter: {key}")

    for key in required_output_keys:
        if key not in params["outputs"]:
            raise MLflowPipelineError(f"Missing output parameter: {key}")

    for key in required_mlflow_keys:
        if key not in params["mlflow"]:
            raise MLflowPipelineError(f"Missing MLflow parameter: {key}")


def save_sentiment_results(df: pd.DataFrame, output_path: str) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info("Sentiment results saved to: %s", output_file)
    return output_file


def run_mlflow_pipeline(params_path: str = "params.yaml") -> pd.DataFrame:
    try:
        params = load_params(params_path)
        validate_mlflow_params(params)

        input_path = params["data"]["input_path"]
        sentiment_results_path = params["data"]["sentiment_results_path"]
        report_path = params["outputs"]["report_path"]
        figure_path = params["outputs"]["figure_path"]
        tracking_uri = params["mlflow"]["tracking_uri"]
        experiment_name = params["mlflow"]["experiment_name"]
        run_name = params["mlflow"]["run_name"]

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("input_path", input_path)
            mlflow.log_param("sentiment_results_path", sentiment_results_path)
            mlflow.log_param(
                "text_columns",
                ",".join(params["pipeline"]["text_columns"]),
            )
            mlflow.log_param(
                "sentiment_input_column",
                params["pipeline"]["sentiment_input_column"],
            )
            mlflow.log_param(
                "vader_threshold_positive",
                params["pipeline"]["vader_threshold_positive"],
            )
            mlflow.log_param(
                "vader_threshold_negative",
                params["pipeline"]["vader_threshold_negative"],
            )
            mlflow.log_param("baseline_model", "VADER")
            mlflow.log_param("main_model", "pysentimiento")

            df = run_data_ingestion(input_path)
            rows_after_ingestion = len(df)

            df = preprocess_dataframe(df)
            rows_after_preprocessing = len(df)

            df = apply_vader_sentiment(df)
            df = apply_pysentimiento_sentiment(df)

            sentiment_results_file = save_sentiment_results(
                df=df,
                output_path=sentiment_results_path,
            )

            agreement_rate = calculate_agreement_rate(df)
            distribution_df = build_sentiment_distribution(df)
            comparison_table = build_model_comparison_table(df)

            report_file = save_evaluation_report(
                agreement_rate=agreement_rate,
                distribution_df=distribution_df,
                comparison_table=comparison_table,
                output_path=report_path,
            )

            figure_file = save_sentiment_distribution_plot(
                distribution_df=distribution_df,
                output_path=figure_path,
            )

            mlflow.log_metric("rows_after_ingestion", rows_after_ingestion)
            mlflow.log_metric("rows_after_preprocessing", rows_after_preprocessing)
            mlflow.log_metric("agreement_rate", agreement_rate)

            vader_distribution = df["vader_sentiment"].value_counts(normalize=True)
            pys_distribution = (
                df["pysentimiento_sentiment"].value_counts(normalize=True)
            )

            for sentiment in ["positive", "neutral", "negative"]:
                mlflow.log_metric(
                    f"vader_{sentiment}_share",
                    float(vader_distribution.get(sentiment, 0.0)),
                )
                mlflow.log_metric(
                    f"pysentimiento_{sentiment}_share",
                    float(pys_distribution.get(sentiment, 0.0)),
                )

            mlflow.log_artifact(str(report_file))
            mlflow.log_artifact(str(figure_file))
            mlflow.log_artifact(str(sentiment_results_file))

            logger.info("MLflow pipeline completed successfully")
            logger.info("Tracking URI: %s", tracking_uri)
            logger.info("Rows after ingestion: %d", rows_after_ingestion)
            logger.info("Rows after preprocessing: %d", rows_after_preprocessing)
            logger.info("Agreement rate: %.4f", agreement_rate)

            return df

    except (DataIngestionError, PreprocessingError, MLflowPipelineError):
        raise

    except Exception as error:
        logger.exception("Unexpected error in MLflow pipeline")
        raise MLflowPipelineError(
            f"Unexpected error in MLflow pipeline: {error}"
        ) from error


def main() -> None:
    run_mlflow_pipeline()


if __name__ == "__main__":
    main()
