"""Train-period-only supervised feature selection from notebooks 2 and 3."""

import hashlib
import json

import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier

from ieee_cis_fraud_detection.validation import time_ordered_indices


def _select_by_gain(importance: pd.Series, fraction: float) -> list[str]:
    if not 0 < fraction <= 1:
        raise ValueError("gain fraction must be between 0 and 1")
    ranked = importance.sort_values(ascending=False, kind="stable")
    total = float(ranked.sum())
    if total <= 0:
        raise ValueError("Feature importance has no positive gain")
    count = int(np.searchsorted((ranked.cumsum() / total).to_numpy(), fraction, side="left")) + 1
    return ranked.head(count).index.tolist()


def select_supervised_features(data: pd.DataFrame, split_options: dict, options: dict):
    """Rank V and ID features using only rows before the evaluation period."""
    if split_options.get("split_strategy") != "time_ordered":
        raise ValueError("Supervised feature selection requires a time_ordered split")
    time_column = split_options["time_column"]
    train_idx, test_idx = time_ordered_indices(data, time_column, split_options["test_size"])
    train = data.loc[train_idx]
    target = train[split_options["target"]]
    positive_count = int((target == 1).sum())
    negative_count = int((target == 0).sum())
    if not positive_count or not negative_count:
        raise ValueError("Feature selection requires both fraud and non-fraud training rows")
    weight = negative_count / positive_count

    v_columns = [col for col in data.columns if col.startswith("V")]
    id_columns = [col for col in data.columns if col.startswith("id_")]
    if not v_columns or not id_columns:
        raise ValueError("Expected V and id_ features for supervised selection")

    v_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=options["v_n_estimators"],
        learning_rate=options["learning_rate"],
        max_depth=options["max_depth"],
        subsample=options["subsample"],
        colsample_bytree=options["colsample_bytree"],
        n_jobs=options["n_jobs"],
        random_state=options["random_state"],
        scale_pos_weight=weight,
        importance_type="gain",
        tree_method="hist",
    )
    v_model.fit(train[v_columns].astype("float32"), target)
    v_importance = pd.Series(v_model.feature_importances_, index=v_columns)
    selected_v = _select_by_gain(v_importance, options["v_gain_fraction"])

    id_data = train[id_columns].copy()
    missing_columns = []
    for col in id_columns:
        if id_data[col].isna().any():
            missing_col = f"{col}_is_missing"
            id_data[missing_col] = id_data[col].isna().astype("int8")
            missing_columns.append(missing_col)
    for col in id_columns:
        if (pd.api.types.is_object_dtype(id_data[col])
                or pd.api.types.is_string_dtype(id_data[col])
                or isinstance(id_data[col].dtype, pd.CategoricalDtype)):
            id_data[col] = id_data[col].astype("object").fillna("__MISSING__").astype("category")

    id_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=options["id_n_estimators"],
        learning_rate=options["learning_rate"],
        max_depth=options["max_depth"],
        subsample=options["subsample"],
        colsample_bytree=options["colsample_bytree"],
        n_jobs=options["n_jobs"],
        random_state=options["random_state"],
        scale_pos_weight=weight,
        importance_type="gain",
        enable_categorical=True,
        tree_method="hist",
    )
    id_model.fit(id_data, target)
    id_importance = pd.Series(id_model.feature_importances_, index=id_data.columns)
    selected_id = _select_by_gain(id_importance, options["id_gain_fraction"])

    report = {
        "method": "XGBoost gain importance fitted on chronological training rows only",
        "xgboost_version": xgboost.__version__,
        "pandas_version": pd.__version__,
        "v_feature_sha256": hashlib.sha256(json.dumps(selected_v, separators=(",", ":")).encode()).hexdigest(),
        "id_feature_sha256": hashlib.sha256(json.dumps(selected_id, separators=(",", ":")).encode()).hexdigest(),
        "split_strategy": "time_ordered",
        "time_column": time_column,
        "test_size": split_options["test_size"],
        "n_selection_rows": len(train_idx),
        "n_held_out_rows": len(test_idx),
        "selection_time_max": float(train[time_column].max()),
        "holdout_time_min": float(data.loc[test_idx, time_column].min()),
        "scale_pos_weight": weight,
        "n_v_candidates": len(v_columns),
        "n_id_candidates": len(id_data.columns),
        "n_id_missing_indicators": len(missing_columns),
        "n_selected_v_features": len(selected_v),
        "n_selected_id_features": len(selected_id),
        "selected_v_features": selected_v,
        "selected_id_features": selected_id,
        "selection_options": options,
    }
    return selected_v, selected_id, report
