FROM python:3.13.18-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    KEDRO_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Match the tested Colab/Linux selector stack and pin the Kedro runtime.
RUN python -m pip install --no-cache-dir \
    numpy==2.4.6 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    xgboost==3.4.1 \
    lightgbm==4.7.0 \
    pyarrow==24.0.0 \
    kedro==1.4.0 \
    'kedro-datasets[pandas-csvdataset,pandas-parquetdataset]==9.5.0'

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY conf ./conf
COPY docker/colab-repro/reproduce_feature_selection.py ./reproduce_feature_selection.py
COPY docker/kedro/stage_selection_artifacts.py ./stage_selection_artifacts.py
COPY docker/kedro/run_staged.sh ./run_staged.sh
RUN chmod +x ./run_staged.sh
RUN mkdir -p data

ENTRYPOINT ["kedro"]
CMD ["run"]
