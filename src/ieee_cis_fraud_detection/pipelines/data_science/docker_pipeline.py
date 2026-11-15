"""Docker model comparison under the Mac's 7 GiB container memory limit."""

from kedro.pipeline import Node, Pipeline

from . import boosted_pipeline, tree_pipeline
from .nodes import (
    combine_feature_groups,
    compare_models_without_baseline,
    evaluate_model,
    evaluate_selected_threshold,
    predict_model,
    select_final_model_without_baseline,
    select_final_threshold_without_baseline,
    tune_model_threshold,
)
from .pipeline import create_split_pipeline


def create_pipeline() -> Pipeline:
    # The LogisticRegression baseline is skipped here because its full-data
    # preprocessing exceeded the Mac's 7 GiB Docker limit. The regular Kedro
    # model_comparison pipeline still includes it.
    return (
        create_split_pipeline()
        + tree_pipeline.create_pipeline()
        + boosted_pipeline.create_branch("lightgbm")
        + boosted_pipeline.create_branch("xgboost")
        + create_finalize_pipeline()
    )


def create_finalize_pipeline() -> Pipeline:
    """Combine already saved branch artifacts without retraining the models."""
    return Pipeline([
            Node(
                func=compare_models_without_baseline,
                inputs=[
                    "tree_model_metrics", "lightgbm_metrics", "xgboost_metrics",
                    "params:model_options", "split_summary", "feature_selection_report",
                ],
                outputs="model_comparison",
                name="compare_docker_models_node",
            ),
            Node(
                func=select_final_model_without_baseline,
                inputs=["tree_model", "lightgbm_model", "xgboost_model", "model_comparison"],
                outputs="final_model",
                name="select_docker_model_node",
            ),
            Node(
                func=select_final_threshold_without_baseline,
                inputs=[
                    "model_comparison", "tree_threshold_report",
                    "lightgbm_threshold_report", "xgboost_threshold_report",
                ],
                outputs="final_model_threshold",
                name="select_docker_threshold_node",
            ),
        ])


def _model_features_pipeline() -> Pipeline:
    return Pipeline([
        Node(
            func=combine_feature_groups,
            inputs=["numeric_model_features", "categorical_model_features"],
            outputs="model_features",
            name="combine_docker_feature_groups_node",
        ),
    ])


def create_lightgbm_branch() -> Pipeline:
    return _model_features_pipeline() + boosted_pipeline.create_branch("lightgbm")


def create_xgboost_branch() -> Pipeline:
    return _model_features_pipeline() + boosted_pipeline.create_branch("xgboost")


def create_saved_evaluation_pipeline() -> Pipeline:
    """Score the already saved tree and LightGBM models after an interrupted run."""
    nodes = []
    for name, features in [("tree", "numeric_model_features"), ("lightgbm", "model_features")]:
        model_name = "tree_model" if name == "tree" else "lightgbm_model"
        predictions_name = "tree_model_predictions" if name == "tree" else "lightgbm_predictions"
        metrics_name = "tree_model_metrics" if name == "tree" else "lightgbm_metrics"
        report_name = "tree_threshold_report" if name == "tree" else "lightgbm_threshold_report"
        nodes.extend([
            Node(
                func=predict_model,
                inputs=[model_name, "X_test", "y_test", features],
                outputs=predictions_name,
                name=f"predict_saved_{name}_node",
            ),
            Node(
                func=evaluate_model,
                inputs=predictions_name,
                outputs=metrics_name,
                name=f"evaluate_saved_{name}_node",
            ),
            Node(
                func=tune_model_threshold,
                inputs=[model_name, "X_train", "y_train", features, "params:model_options", "params:threshold_options"],
                outputs=f"{name}_inner_threshold_report",
                name=f"tune_saved_{name}_threshold_node",
            ),
            Node(
                func=evaluate_selected_threshold,
                inputs=[f"{name}_inner_threshold_report", predictions_name, "params:threshold_options"],
                outputs=report_name,
                name=f"evaluate_saved_{name}_threshold_node",
            ),
        ])
    return _model_features_pipeline() + Pipeline(nodes)
