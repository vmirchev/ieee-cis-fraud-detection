"""Notebook-equivalent LightGBM and XGBoost training nodes."""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from xgboost import XGBClassifier

from .nodes import create_model_metadata
from .transformers import RareCategoryGrouper


def _numeric_float32(frame):
    return frame.astype("float32", copy=False)


def _preprocessor(numeric_features: list[str], categorical_features: list[str], rare_min_frequency: float, low_memory: bool = False) -> ColumnTransformer:
    numeric_steps = []
    if low_memory:
        numeric_steps.append(("float32", FunctionTransformer(_numeric_float32)))
    numeric_steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=True, copy=not low_memory)))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline([
        ("rare", RareCategoryGrouper(min_frequency=rare_min_frequency)),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            dtype=np.float32 if low_memory else np.float64,
        )),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_features),
        ("cat", categorical, categorical_features),
    ], remainder="drop")


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    parameters: dict,
) -> Pipeline:
    features = numeric_features + categorical_features
    model = Pipeline([
        ("preprocess", _preprocessor(numeric_features, categorical_features, parameters["rare_min_frequency"], parameters.get("numeric_dtype") == "float32")),
        ("model", LGBMClassifier(
            objective=parameters["objective"],
            metric=parameters["metric"],
            n_estimators=parameters["n_estimators"],
            learning_rate=parameters["learning_rate"],
            num_leaves=parameters["num_leaves"],
            max_depth=parameters["max_depth"],
            random_state=parameters["random_state"],
            n_jobs=parameters["n_jobs"],
            class_weight=parameters["class_weight"],
            verbosity=parameters.get("verbosity", -1),
        )),
    ])
    return model.fit(X_train[features], y_train.iloc[:, 0])


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    parameters: dict,
) -> Pipeline:
    target = y_train.iloc[:, 0]
    positive_count = int((target == 1).sum())
    negative_count = int((target == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("XGBoost training requires both fraud and non-fraud rows")
    scale_pos_weight = negative_count / positive_count
    features = numeric_features + categorical_features
    model = Pipeline([
        ("preprocess", _preprocessor(numeric_features, categorical_features, parameters["rare_min_frequency"], parameters.get("numeric_dtype") == "float32")),
        ("model", XGBClassifier(
            objective=parameters["objective"],
            eval_metric=parameters["eval_metric"],
            n_estimators=parameters["n_estimators"],
            learning_rate=parameters["learning_rate"],
            max_depth=parameters["max_depth"],
            subsample=parameters["subsample"],
            colsample_bytree=parameters["colsample_bytree"],
            min_child_weight=parameters["min_child_weight"],
            reg_lambda=parameters["reg_lambda"],
            random_state=parameters["random_state"],
            n_jobs=parameters["n_jobs"],
            tree_method=parameters["tree_method"],
            scale_pos_weight=scale_pos_weight,
        )),
    ])
    return model.fit(X_train[features], target)


def create_boosted_model_metadata(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    model: Pipeline,
    parameters: dict,
    metrics: dict,
) -> dict:
    metadata = create_model_metadata(
        X_train, X_test, y_train, numeric_features + categorical_features, parameters, metrics,
    )
    metadata["numeric_features"] = numeric_features
    metadata["categorical_features"] = categorical_features
    metadata["preprocessing"] = {
        "numeric": "median imputation with missing indicators",
        "categorical": "rare category grouping, constant imputation, ordinal encoding with unknown=-1",
    }
    if isinstance(model.named_steps["model"], XGBClassifier):
        metadata["effective_scale_pos_weight"] = model.named_steps["model"].scale_pos_weight
    return metadata
