#!/bin/sh
set -eu

mkdir -p /app/data/03_primary /app/data/08_reporting
python /app/reproduce_feature_selection.py \
  --data-dir /app/data/01_raw \
  --reducer project \
  --group both \
  --n-jobs=-1 \
  > /app/data/08_reporting/docker_selection_result.json
python /app/stage_selection_artifacts.py
kedro run --env docker --pipelines docker_staged_data
kedro run --env docker --pipelines docker_split
kedro run --env docker --pipelines docker_tree_only
kedro run --env docker --pipelines docker_lightgbm_only
kedro run --env docker --pipelines docker_xgboost_only
exec kedro run --env docker --pipelines docker_finalize
