from kedro.pipeline import Node, Pipeline

from .nodes import evaluate_model, create_model_metadata, predict_model, tune_model_threshold, evaluate_selected_threshold, train_tree_model

def create_tree_pipeline() -> Pipeline:
    return Pipeline([
        Node(
            func=train_tree_model,
            inputs=["X_train", "y_train", "numeric_model_features", "params:tree_model_options"],
            outputs="tree_model",
            name="train_tree_model_node",
            tags=["tree", "train"]
        ),
        Node(
            func=predict_model,
            inputs=["tree_model", "X_test", "y_test", "numeric_model_features"],
            outputs="tree_model_predictions",
            name="predict_tree_model_node",
            tags=["tree", "predict"]
        ),
        Node(
            func=evaluate_model,
            inputs="tree_model_predictions",
            outputs="tree_model_metrics",
            name="evaluate_tree_model_node",
            tags=["tree", "evaluate"]
        )]
    )

def create_tree_reporting_pipeline() -> Pipeline:
    return Pipeline(
        [
            Node(
                func=create_model_metadata,
                inputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "numeric_model_features",
                    "params:tree_model_options",
                    "tree_model_metrics",
                ],
                outputs="tree_model_metadata",
                name="create_tree_model_metadata_node",
                tags=["tree", "reporting"]
            ),
        ]
    )

def threshold_tree_tuning_pipeline() -> Pipeline:
    return Pipeline([
            Node(
                func=tune_model_threshold,
                inputs=["tree_model", "X_train", "y_train", "numeric_model_features", "params:model_options", "params:threshold_options"],
                outputs="tree_inner_threshold_report",
                name="tune_tree_threshold_node",
                tags=["tree", "threshold"]
            ),
            Node(
                func=evaluate_selected_threshold,
                inputs=["tree_inner_threshold_report", "tree_model_predictions", "params:threshold_options"],
                outputs="tree_threshold_report",
                name="evaluate_tree_selected_threshold_node",
                tags=["tree", "threshold"],
            ),
    ])

def create_pipeline(**kwargs) -> Pipeline:
    return (create_tree_pipeline() +
            create_tree_reporting_pipeline() +
            threshold_tree_tuning_pipeline())
