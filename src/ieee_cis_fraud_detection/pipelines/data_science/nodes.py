import pandas as pd
import numpy as np
from pandas import CategoricalDtype
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .transformers import RareCategoryGrouper
from ieee_cis_fraud_detection.validation import time_ordered_indices


def _split_random_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify = y if parameters["stratify"] else None

    return train_test_split(
        X,
        y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=stratify,
    )


def _split_time_ordered(
    X: pd.DataFrame,
    y: pd.Series,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    time_column = parameters["time_column"]
    if time_column not in X.columns:
        raise KeyError(time_column)

    train_idx, test_idx = time_ordered_indices(X, time_column, parameters["test_size"])

    return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]


def split_data(data: pd.DataFrame, parameters: dict):
    target = parameters["target"]
    drop_columns = parameters["drop_columns"]

    split_strategy = parameters.get("split_strategy", "random_stratified")
    time_column = parameters.get("time_column")
    split_required_columns = [time_column] if split_strategy == "time_ordered" else []
    model_drop_columns = [
        column for column in drop_columns if column not in split_required_columns
    ]

    X = data.drop(columns=[target, *model_drop_columns], errors="ignore")
    y = data[target]

    if split_strategy == "random_stratified":
        X_train, X_test, y_train, y_test = _split_random_stratified(X, y, parameters)
    elif split_strategy == "time_ordered":
        X_train, X_test, y_train, y_test = _split_time_ordered(X, y, parameters)
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")

    X_train = X_train.drop(columns=split_required_columns, errors="ignore")
    X_test = X_test.drop(columns=split_required_columns, errors="ignore")

    return (
        X_train,
        X_test,
        y_train.to_frame(name=target),
        y_test.to_frame(name=target),
    )

def create_split_summary(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
  ) -> dict:
    target = y_train.columns[0]

    return {
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "n_features": X_train.shape[1],
        "train_fraud_rate": float(y_train[target].mean()),
        "test_fraud_rate": float(y_test[target].mean()),
    }

def create_numeric_feature_list(X_train: pd.DataFrame) -> list[str]:
    return X_train.select_dtypes(include="number").columns.tolist()


def _is_categorical_like(series: pd.Series) -> bool:
    return (
        is_object_dtype(series)
        or isinstance(series.dtype, CategoricalDtype)
        or is_string_dtype(series)
        or is_bool_dtype(series)
    )


def create_feature_groups(
    X_train: pd.DataFrame,
    parameters: dict,
) -> tuple[list[str], list[str]]:
    id_like_numeric = set(parameters["id_like_numeric"])
    known_categorical = set(parameters["known_categorical"])
    exclude_from_features = set(parameters["exclude_from_features"])

    numeric_features = []
    categorical_features = []

    for column in X_train.columns:
        if column in exclude_from_features:
            continue

        if column in id_like_numeric or column in known_categorical:
            categorical_features.append(column)
            continue

        if _is_categorical_like(X_train[column]):
            categorical_features.append(column)
        elif is_numeric_dtype(X_train[column]):
            numeric_features.append(column)

    return numeric_features, categorical_features


def combine_feature_groups(
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    return numeric_features + categorical_features


def train_model(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        numeric_features: list[str],
        categorical_features: list[str],
        parameters: dict,
  ) -> Pipeline:
    target = y_train.columns[0]
    y_train_series = y_train[target]
    model_features = numeric_features + categorical_features

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "rare",
                RareCategoryGrouper(
                    min_frequency=parameters["rare_min_frequency"],
                ),
            ),
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Missing"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    random_state=parameters["random_state"],
                    class_weight=parameters["class_weight"],
                    max_iter=parameters["max_iter"],
                    solver=parameters.get("solver", "lbfgs"),
                    tol=parameters.get("tol", 1e-4),
                ),
            ),
        ]
    )

    model.fit(X_train[model_features], y_train_series)

    return model

def train_tree_model(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        model_features: list[str],
        parameters: dict
) -> RandomForestClassifier:
    target = y_train.columns[0]
    y_train_series = y_train[target]

    model = RandomForestClassifier(
        n_estimators=parameters["n_estimators"],
        max_depth=parameters["max_depth"],
        random_state=parameters["random_state"],
        class_weight=parameters["class_weight"],
    )

    model.fit(X_train[model_features], y_train_series)

    return model

def predict_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model_features: list[str],
) -> pd.DataFrame:
    target = y_test.columns[0]

    return pd.DataFrame(
        {
            target: y_test[target].values,
            "y_score": model.predict_proba(X_test[model_features])[:, 1],
        },
        index=X_test.index,
    )


def evaluate_model(predictions: pd.DataFrame) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(predictions["isFraud"], predictions["y_score"]),
        "pr_auc": average_precision_score(predictions["isFraud"], predictions["y_score"]),
    }


def create_model_metadata(
      X_train: pd.DataFrame,
      X_test: pd.DataFrame,
      y_train: pd.DataFrame,
      model_features: list[str],
      model_parameters: dict,
      metrics: dict,
  ) -> dict:
    return {
        "model_name": model_parameters["model_name"],
        "target": y_train.columns[0],
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "n_features": len(model_features),
        "model_features": model_features,
        "model_parameters": model_parameters,
        "metrics": metrics,
    }

def tune_threshold(predictions: pd.DataFrame, parameters: dict) -> dict:
    target = parameters["target"]
    thresholds = parameters.get("thresholds")
    if thresholds is None:
        start = parameters["threshold_start"]
        stop = parameters["threshold_stop"]
        step = parameters["threshold_step"]
        thresholds = np.round(np.arange(start, stop + step / 2, step), 2).tolist()

    rows = []
    for threshold in thresholds:
        y_pred = predictions["y_score"] >= threshold
        matrix = confusion_matrix(predictions[target], y_pred, labels=[0, 1])

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(predictions[target], y_pred, zero_division=0),
                "recall": recall_score(predictions[target], y_pred, zero_division=0),
                "f1": f1_score(predictions[target], y_pred, zero_division=0),
                "false_positives": int(matrix[0, 1]),
                "false_negatives": int(matrix[1, 0]),
            }
        )

    best = max(rows, key=lambda row: row["f1"])

    return {
        "thresholds": rows,
        "best_threshold_by_f1": best,
    }


def tune_model_threshold(
    model,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_features: list[str],
    split_options: dict,
    threshold_options: dict,
) -> dict:
    """Choose a threshold within the outer training period, as in notebook 4."""
    inner_fraction = threshold_options.get("inner_validation_fraction", 0.2)
    if not 0 < inner_fraction < 1:
        raise ValueError("inner_validation_fraction must be between 0 and 1")
    target = threshold_options["target"]
    if split_options["split_strategy"] == "time_ordered":
        # split_data already returns X_train in TransactionDT order.
        cutoff = int(len(X_train) * (1 - inner_fraction))
        fit_idx, val_idx = X_train.index[:cutoff], X_train.index[cutoff:]
    else:
        fit_idx, val_idx = train_test_split(
            X_train.index,
            test_size=inner_fraction,
            random_state=split_options["random_state"],
            stratify=y_train[target] if split_options.get("stratify") else None,
        )
    if len(fit_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Inner threshold split needs both fitting and validation rows")

    inner_model = clone(model)
    inner_model.fit(X_train.loc[fit_idx, model_features], y_train.loc[fit_idx, target])
    scores = inner_model.predict_proba(X_train.loc[val_idx, model_features])[:, 1]
    predictions = pd.DataFrame(
        {target: y_train.loc[val_idx, target], "y_score": scores}, index=val_idx
    )
    report = tune_threshold(predictions, threshold_options)
    report.update({
        "selection_period": "inner_training_validation",
        "n_inner_fit_rows": len(fit_idx),
        "n_inner_validation_rows": len(val_idx),
    })
    return report


def evaluate_selected_threshold(
    threshold_report: dict,
    holdout_predictions: pd.DataFrame,
    threshold_options: dict,
) -> dict:
    """Report holdout performance after the inner threshold is fixed."""
    target = threshold_options["target"]
    threshold = threshold_report["best_threshold_by_f1"]["threshold"]
    predicted = holdout_predictions["y_score"] >= threshold
    actual = holdout_predictions[target]
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    return {
        **threshold_report,
        "holdout_at_selected_threshold": {
            "threshold": threshold,
            "precision": precision_score(actual, predicted, zero_division=0),
            "recall": recall_score(actual, predicted, zero_division=0),
            "f1": f1_score(actual, predicted, zero_division=0),
            "false_positives": int(matrix[0, 1]),
            "false_negatives": int(matrix[1, 0]),
        },
    }

def compare_models(
    baseline_metrics: dict,
    tree_metrics: dict,
    lightgbm_metrics: dict,
    xgboost_metrics: dict,
    split_options: dict,
    split_summary: dict,
    selection_report: dict,
) -> dict:
    if (selection_report["split_strategy"] != split_options["split_strategy"]
            or selection_report["time_column"] != split_options["time_column"]
            or selection_report["test_size"] != split_options["test_size"]
            or selection_report["n_selection_rows"] != split_summary["n_train_rows"]
            or selection_report["n_held_out_rows"] != split_summary["n_test_rows"]):
        raise ValueError("Feature-selection boundary does not match model evaluation split")
    rows = {
        "tree": tree_metrics,
        "lightgbm": lightgbm_metrics,
        "xgboost": xgboost_metrics,
    }
    if baseline_metrics is not None:
        rows["baseline"] = baseline_metrics

    best_by_pr_auc = max(
        rows,
        key=lambda model_name: rows[model_name]["pr_auc"],
    )

    best_by_roc_auc = max(
        rows,
        key=lambda model_name: rows[model_name]["roc_auc"],
    )

    return {
        "selection_metric": "pr_auc",
        "split_options": split_options,
        "split_summary": split_summary,
        "feature_selection": selection_report,
        "models": rows,
        "best_by_pr_auc": best_by_pr_auc,
        "best_by_roc_auc": best_by_roc_auc,
    }


def select_final_model(baseline_model, tree_model, lightgbm_model, xgboost_model, comparison: dict):
    """Persist the model selected by the shared holdout comparison."""
    return {
        "baseline": baseline_model,
        "tree": tree_model,
        "lightgbm": lightgbm_model,
        "xgboost": xgboost_model,
    }[comparison["best_by_pr_auc"]]


def select_final_threshold(
    comparison: dict,
    baseline_report: dict,
    tree_report: dict,
    lightgbm_report: dict,
    xgboost_report: dict,
) -> dict:
    reports = {
        "baseline": baseline_report,
        "tree": tree_report,
        "lightgbm": lightgbm_report,
        "xgboost": xgboost_report,
    }
    winner = comparison["best_by_pr_auc"]
    report = reports[winner]
    return {
        "selected_model": winner,
        "selection_metric": comparison["selection_metric"],
        "threshold_selection_period": report["selection_period"],
        "selected_threshold": report["best_threshold_by_f1"]["threshold"],
        "inner_validation_at_selected_threshold": report["best_threshold_by_f1"],
        "holdout_at_selected_threshold": report["holdout_at_selected_threshold"],
    }


def compare_models_without_baseline(
    tree_metrics: dict,
    lightgbm_metrics: dict,
    xgboost_metrics: dict,
    split_options: dict,
    split_summary: dict,
    selection_report: dict,
) -> dict:
    return compare_models(
        None, tree_metrics, lightgbm_metrics, xgboost_metrics,
        split_options, split_summary, selection_report,
    )


def select_final_model_without_baseline(
    tree_model, lightgbm_model, xgboost_model, comparison: dict,
):
    return select_final_model(None, tree_model, lightgbm_model, xgboost_model, comparison)


def select_final_threshold_without_baseline(
    comparison: dict, tree_report: dict, lightgbm_report: dict, xgboost_report: dict,
) -> dict:
    return select_final_threshold(comparison, None, tree_report, lightgbm_report, xgboost_report)
