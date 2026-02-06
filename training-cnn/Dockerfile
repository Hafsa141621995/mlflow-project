FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch torchvision \
    mlflow

COPY train.py .

CMD ["python", "train.py"]
