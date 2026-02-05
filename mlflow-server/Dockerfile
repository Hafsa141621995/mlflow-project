FROM python:3.10-slim

# Installer les certificats SSL
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates

# Installer MLflow en bypassant la vérification SSL
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org mlflow

EXPOSE 5000

CMD mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "*" 