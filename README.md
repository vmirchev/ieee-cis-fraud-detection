# IEEE-CIS Fraud Detection

[![Built with Kedro](https://img.shields.io/badge/built_with-Kedro-ffc900?logo=kedro)](https://kedro.org)

**Built with Kedro**, this project turns five Google Colab notebooks on the IEEE-CIS fraud detection data into a reproducible modeling pipeline. It runs feature selection, preprocessing, training, and evaluation in a defined order and saves the results for inspection.

The question is simple: **given a transaction, how well can we identify fraud?** The data and the answer are less simple. Fraud is rare, many fields are missing, and transactions arrive over time. Those details shaped the split, the features, and the metrics used here.

The [notebook experiments and findings](docs/notebook_experiments.md) explain the choices behind the pipeline. The [Docker run guide](docker/kedro/README.md) shows how I reproduced the main Colab results on Linux within this Mac's memory limit.

## What the project does

1. Joins the transaction and identity files and reduces their memory use.
2. Reserves the latest 20% of transactions as a chronological holdout.
3. Selects V and ID features using **only the earlier 80%**.
4. Builds time, amount, and identity missingness features.
5. Trains logistic regression, random forest, LightGBM, and XGBoost on the same split in the regular Kedro pipeline.
6. Compares the models by PR-AUC, chooses a decision threshold on an inner training-period slice, and saves the models and reports.

The Docker pipeline runs the same feature and model workflow in smaller stages. It skips logistic regression because its full-data fit exceeded the 7 GiB container limit on my Mac. The Docker comparison therefore includes random forest, LightGBM, and XGBoost.

## Results

The completed Docker run used 472,432 earlier transactions for training and 118,108 later transactions for the holdout. Linux feature selection retained 229 V features and 28 ID features, matching the Colab lists exactly.

| Model | Holdout PR-AUC | Holdout ROC-AUC |
| --- | ---: | ---: |
| Random forest | 0.432927 | 0.865198 |
| LightGBM | **0.535803** | 0.904712 |
| XGBoost | 0.534329 | **0.909574** |

LightGBM won by PR-AUC. Its decision threshold of **0.79** was selected on an inner chronological validation slice of the training period. At that threshold, holdout precision was **0.613858** and recall was **0.449065**. The LightGBM and XGBoost scores, selected model, and threshold match the downloaded Colab notebook 4 results.

PR-AUC is the main comparison metric because fraud is uncommon; ROC-AUC is included for context. The threshold is picked without using holdout labels. The holdout has, however, been inspected during this project's development, so these scores compare the current candidates but are not an untouched estimate of future performance. A new, later labeled period would provide that final check.

## Why feature selection changed

The early notebooks used random splits while choosing features. That let transactions from the later evaluation period influence which features were kept. I changed the process so feature selection happens strictly within the earlier 80% of transactions, before the holdout is evaluated.

Two XGBoost selectors rank the V and ID feature groups separately by gain. For each group, the pipeline keeps the smallest ranked set that reaches **95% of total positive gain**. This rule replaces fixed feature counts chosen after looking at earlier results. ID missingness flags and categorical levels used by the selectors are also determined from training-period rows. The generated lists and their provenance are saved in `data/03_primary/` and `data/08_reporting/` when the pipeline runs; generated data is excluded from Git.

Selection can vary slightly by platform. Colab and the Linux Docker run produced the same 229 V / 28 ID lists; a local macOS run produced 230 V / 31 ID lists. The [reproduction notes](docker/colab-repro/DEBUGGING.md) document that investigation. Model scores should be read alongside the feature lists and environment that produced them.

## Run it

Download the IEEE-CIS competition data from Kaggle and place these files in `data/01_raw/`:

```text
train_transaction.csv
train_identity.csv
```

The data, generated feature tables, and trained models are intentionally excluded from this repository.

For the tested Docker workflow, follow [docker/kedro/README.md](docker/kedro/README.md). From the repository root, the main commands are:

```bash
docker build --platform linux/amd64 -f docker/kedro/Dockerfile -t ieee-kedro .
mkdir -p docker/kedro/output
docker run --rm --platform linux/amd64 --cpus=2 --memory=7g \
  --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 \
  --env MALLOC_ARENA_MAX=2 \
  --mount "type=bind,src=$PWD/docker/kedro/output,dst=/app/data" \
  --mount "type=bind,src=$PWD/data/01_raw,dst=/app/data/01_raw,readonly" \
  --entrypoint /app/run_staged.sh ieee-kedro
```

To use the regular Kedro pipeline, install the project requirements and run:

```bash
pip install -r requirements.txt
kedro run
```

The regular run includes logistic regression and needs more memory than the 7 GiB Docker profile. On macOS, LightGBM and XGBoost also need OpenMP. If `libomp.dylib` is missing, install `libomp` or point `DYLD_LIBRARY_PATH` at your OpenMP library before running Kedro.

The five notebooks can be rerun in order in fresh Colab runtimes. They expect the `KAGGLE_API_TOKEN` Colab secret and a Drive folder at `/My Drive/ML/Fraud Detection/Data/Artifacts`. Notebooks 2 and 3 select features; notebook 4 runs the fixed-parameter model comparison; notebook 5 explores Optuna tuning. **Notebook 5's search is not part of the Kedro model comparison.** The [experiment notes](docs/notebook_experiments.md) explain the steps and reasoning.

## How the code is organized

| Location | Purpose |
| --- | --- |
| `notebooks/` | Colab exploration and saved outputs |
| `src/ieee_cis_fraud_detection/pipelines/data_processing/` | Join, reduce, select, and engineer features |
| `src/ieee_cis_fraud_detection/pipelines/data_science/` | Split, preprocess, train, evaluate, and compare models |
| `conf/base/` | Kedro datasets and regular pipeline parameters |
| `conf/docker/` and `docker/kedro/` | Memory-conscious Linux run and its instructions |
| `docker/colab-repro/` | Feature-selection reproduction and debugging notes |
| `tests/` | Pipeline and model checks on synthetic data |

Kedro saves the selected feature lists, split summary, feature-selection report, model metrics, threshold reports, comparison, and final model under `data/`. The default pipeline includes all four candidates; the Docker stages save their results under `docker/kedro/output/`.

Run the focused tests with:

```bash
pip install -e ".[dev]"
python -m pytest tests/pipelines/data_processing/test_feature_selection.py tests/pipelines/data_science -q
```

The full suite includes a run over the Kaggle data and takes substantially longer.

## Scope

This is a reproducible model comparison, not a Kaggle test-set submission or a deployed fraud service. The selected model has not been refit on every available labeled transaction. Earlier experimental scores and the history of changes are retained in the notebooks and [experiment notes](docs/notebook_experiments.md); the table above describes the completed Docker run with the current preprocessing and feature lists.
