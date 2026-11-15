"""LightGBM and XGBoost branches using the shared model split."""

from kedro.pipeline import Node, Pipeline

from .boosted_nodes import create_boosted_model_metadata, train_lightgbm_model, train_xgboost_model
from .nodes import evaluate_model, predict_model, tune_model_threshold, evaluate_selected_threshold


def create_branch(name: str) -> Pipeline:
    if name not in {"lightgbm", "xgboost"}:
        raise ValueError(f"Unknown boosted model: {name}")
    train = train_lightgbm_model if name == "lightgbm" else train_xgboost_model
    return Pipeline([
        Node(
            func=train,
            inputs=["X_train", "y_train", "numeric_model_features", "categorical_model_features", f"params:{name}_model_options"],
            outputs=f"{name}_model",
            name=f"train_{name}_model_node",
            tags=[name, "train"],
        ),
        Node(
            func=predict_model,
            inputs=[f"{name}_model", "X_test", "y_test", "model_features"],
            outputs=f"{name}_predictions",
            name=f"predict_{name}_model_node",
            tags=[name, "predict"],
        ),
        Node(
            func=evaluate_model,
            inputs=f"{name}_predictions",
            outputs=f"{name}_metrics",
            name=f"evaluate_{name}_model_node",
            tags=[name, "evaluate"],
        ),
        Node(
            func=create_boosted_model_metadata,
            inputs=[
                "X_train", "X_test", "y_train", "numeric_model_features", "categorical_model_features",
                f"{name}_model", f"params:{name}_model_options", f"{name}_metrics",
            ],
            outputs=f"{name}_model_metadata",
            name=f"create_{name}_model_metadata_node",
            tags=[name, "reporting"],
        ),
        Node(
            func=tune_model_threshold,
            inputs=[f"{name}_model", "X_train", "y_train", "model_features", "params:model_options", "params:threshold_options"],
            outputs=f"{name}_inner_threshold_report",
            name=f"tune_{name}_threshold_node",
            tags=[name, "threshold"],
        ),
        Node(
            func=evaluate_selected_threshold,
            inputs=[f"{name}_inner_threshold_report", f"{name}_predictions", "params:threshold_options"],
            outputs=f"{name}_threshold_report",
            name=f"evaluate_{name}_selected_threshold_node",
            tags=[name, "threshold"],
        ),
    ])
