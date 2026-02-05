FROM python:3.10-slim

# Fix SSL (important en réseau école/entreprise)
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_TRUSTED_HOST=pypi.org
ENV PIP_TRUSTED_HOST=pypi.python.org
ENV PIP_TRUSTED_HOST=files.pythonhosted.org

RUN pip install --trusted-host pypi.org \
                --trusted-host pypi.python.org \
                --trusted-host files.pythonhosted.org \
                mlflow scikit-learn

WORKDIR /app
COPY train.py train.py

CMD ["python", "train.py"]
