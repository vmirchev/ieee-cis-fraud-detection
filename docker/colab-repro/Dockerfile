FROM python:3.13.18-slim-bookworm

ENV PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir \
    numpy==2.4.6 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    xgboost==3.4.1

COPY reproduce_feature_selection.py /app/reproduce_feature_selection.py
WORKDIR /workspace

ENTRYPOINT ["python", "/app/reproduce_feature_selection.py"]
