# Kedro on Docker Desktop

The Linux image uses Python 3.13.18, NumPy 2.4.6, pandas 2.2.3,
scikit-learn 1.6.1, XGBoost 3.4.1, Kedro 1.4.0, and LightGBM 4.7.0.
On this 16 GB Intel Mac, Docker Desktop has about 7.75 GiB allocated.
The full-width default Kedro pipeline exceeded a 7 GiB container limit,
so this profile runs memory-conscious, restartable stages.

The Docker profile **skips the logistic regression baseline** because its
full-data preprocessing exceeded that memory limit. The regular Kedro and
Colab workflows still include it. The Docker comparison contains random
forest, LightGBM, and XGBoost. XGBoost uses a float32 preprocessing matrix
in this Docker profile to reduce peak memory; its measured holdout scores
matched Colab notebook 4 exactly.

The raw CSVs are mounted read-only. All Docker results are written under
`docker/kedro/output/`, which is ignored by Git, so existing local Kedro
artifacts remain untouched.

## Build and run from scratch

From the repository root, with Docker Desktop running and the two Kaggle
training CSVs in `data/01_raw/`:

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

`run_staged.sh` selects V and ID features in separate narrow reads, builds
the selected feature table, creates the time split, runs each model branch in
a separate Kedro process, and saves the final comparison and selected
threshold. If a stage is interrupted, its earlier artifacts remain in the
output directory. To resume one stage, use the same `docker run` options above,
omit `--entrypoint /app/run_staged.sh`, and append
`ieee-kedro run --env docker --pipelines docker_xgboost_only` (replace the
pipeline name as needed). Available stages are `docker_staged_data`,
`docker_split`, `docker_tree_only`, `docker_lightgbm_only`,
`docker_xgboost_only`, and `docker_finalize`.

## Completed run: 13 November 2026

The Linux selector produced **229 V and 28 ID features**. Their ordered-list
SHA-256 hashes exactly matched the Colab reproduction:

- V: `76a200aa310238b41f3b9e6de163efa8f5334905e787b117ceb227eb2a23435a`
- ID: `938d48a18eb14258a0dd28493ad2547bdf1a709750ec484edb9f86479581c02f`

The modeling split had 472,432 training rows, 118,108 holdout rows, and 314
input columns (280 numeric, 34 categorical).

| Model | Holdout PR-AUC | Holdout ROC-AUC |
| --- | ---: | ---: |
| Random forest | 0.432927 | 0.865198 |
| LightGBM | **0.535803** | 0.904712 |
| XGBoost | 0.534329 | **0.909574** |

LightGBM was selected by PR-AUC. Its threshold of **0.79** was chosen on the
inner training-period validation slice; on the later holdout it gave
precision **0.613858** and recall **0.449065**. LightGBM and XGBoost scores,
the winning model, and this threshold match the downloaded Colab notebook 4.
The final artifacts are under `docker/kedro/output/06_models/`,
`docker/kedro/output/07_model_output/`, and `docker/kedro/output/08_reporting/`.
As in the notebooks, the previously examined holdout is an exploratory
comparison rather than a fresh untouched estimate.
