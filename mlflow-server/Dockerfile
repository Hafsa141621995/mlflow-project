FROM python:3.10-slim

WORKDIR /mlflow

RUN apt-get update && \
    apt-get install -y ca-certificates && \
    update-ca-certificates

RUN pip install --upgrade pip && \
    pip install mlflow azureml-mlflow

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 0.0.0.0 \
    --port 5000