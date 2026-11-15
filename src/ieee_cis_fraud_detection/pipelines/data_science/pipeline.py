from kedro.pipeline import Node, Pipeline

from . import boosted_pipeline, tree_pipeline
from .nodes import combine_feature_groups, create_feature_groups, evaluate_model, split_data, train_model, create_split_summary, create_model_metadata, predict_model, tune_model_threshold, evaluate_selected_threshold, compare_models, select_final_model, select_final_threshold

def create_split_pipeline() -> Pipeline:
    return Pipeline([
        Node(
            func=split_data,
            inputs=["feature_engineered_train", "params:model_options"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node",
            tags=["split"]
        ),
        Node(
            func=create_split_summary,
            inputs=["X_train", "X_test", "y_train", "y_test"],
            outputs="split_summary",
            tags=["split", "reporting"]
        ),
        Node(
            func=create_feature_groups,
            inputs=["X_train", "params:feature_grouping"],
            outputs=["numeric_model_features", "categorical_model_features"],
            name="create_feature_groups_node",
            tags=["split"],
        ),
        Node(
            func=combine_feature_groups,
            inputs=["numeric_model_features", "categorical_model_features"],
            outputs="model_features",
            name="combine_feature_groups_node",
            tags=["split"],
        )]
    )

# The Docker-specific comparison skips this linear baseline because its
# full-data fit exceeded the Mac's 7 GiB container memory limit. Colab and
# the regular Kedro comparison still run it.
def create_baseline_pipeline() -> Pipeline:
    return Pipeline([
        Node(
            func=train_model,
            inputs=[
                "X_train",
                "y_train",
                "numeric_model_features",
                "categorical_model_features",
                "params:baseline_model_options",
            ],
            outputs="baseline_model",
            name="train_baseline_model_node",
            tags=["baseline", "train"]
        ),
        Node(
            func=predict_model,
            inputs=["baseline_model", "X_test", "y_test", "model_features"],
            outputs="baseline_predictions",
            name="predict_baseline_model_node",
            tags=["baseline", "predict"]
        ),
        Node(
            func=evaluate_model,
            inputs="baseline_predictions",
            outputs="baseline_metrics",
            name="evaluate_baseline_model_node",
            tags=["baseline", "evaluate"]
        )]
    )

def create_reporting_pipeline() -> Pipeline:
    return Pipeline(
        [
            Node(
                func=create_model_metadata,
                inputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "model_features",
                    "params:baseline_model_options",
                    "baseline_metrics",
                ],
                outputs="baseline_model_metadata",
                name="create_baseline_model_metadata_node",
                tags=["baseline", "reporting"]
            ),
        ]
    )

def threshold_tuning_pipeline() -> Pipeline:
    return Pipeline([
            Node(
                func=tune_model_threshold,
                inputs=["baseline_model", "X_train", "y_train", "model_features", "params:model_options", "params:threshold_options"],
                outputs="baseline_inner_threshold_report",
                name="tune_baseline_threshold_node",
                tags=["baseline", "threshold"]
            ),
            Node(
                func=evaluate_selected_threshold,
                inputs=["baseline_inner_threshold_report", "baseline_predictions", "params:threshold_options"],
                outputs="baseline_threshold_report",
                name="evaluate_baseline_selected_threshold_node",
                tags=["baseline", "threshold"],
            ),
    ])

def create_tree_model_pipeline(**kwargs) -> Pipeline:
    return (create_split_pipeline() +
            tree_pipeline.create_pipeline())


def create_lightgbm_model_pipeline(**kwargs) -> Pipeline:
    return create_split_pipeline() + boosted_pipeline.create_branch("lightgbm")


def create_xgboost_model_pipeline(**kwargs) -> Pipeline:
    return create_split_pipeline() + boosted_pipeline.create_branch("xgboost")

def create_pipeline(**kwargs) -> Pipeline:
    return (create_split_pipeline() +
            create_baseline_pipeline() +
            create_reporting_pipeline() +
            threshold_tuning_pipeline())

def create_model_comparison_pipeline(**kwargs) -> Pipeline:
    return (create_pipeline() + tree_pipeline.create_pipeline()
        + boosted_pipeline.create_branch("lightgbm")
        + boosted_pipeline.create_branch("xgboost")
        + Pipeline([
            Node(
                func=compare_models,
                inputs=["baseline_metrics", "tree_model_metrics", "lightgbm_metrics", "xgboost_metrics", "params:model_options", "split_summary", "feature_selection_report"],
                outputs="model_comparison",
                name="model_comparison_node",
                tags=["comparison"]
            ),
            Node(
                func=select_final_model,
                inputs=["baseline_model", "tree_model", "lightgbm_model", "xgboost_model", "model_comparison"],
                outputs="final_model",
                name="select_final_model_node",
                tags=["comparison"],
            ),
            Node(
                func=select_final_threshold,
                inputs=["model_comparison", "baseline_threshold_report", "tree_threshold_report", "lightgbm_threshold_report", "xgboost_threshold_report"],
                outputs="final_model_threshold",
                name="select_final_threshold_node",
                tags=["comparison", "threshold"],
            ),
    ]))
