# MLOps with MLflow

## Objective

This project uses MLflow to track the execution of the sentiment analysis pipeline locally.

Since the project uses pre-trained sentiment classifiers, MLflow is used for experiment tracking rather than for training and registering a newly trained model.

---

## MLflow Tracking URI

The local MLflow tracking URI is:

```text
http://127.0.0.1:5000


Running the MLflow Pipeline

Start the MLflow server:

Bash
mlflow ui --host 127.0.0.1 --port 5000

In another terminal, run:

Bash
python src/mlflow_pipeline.py