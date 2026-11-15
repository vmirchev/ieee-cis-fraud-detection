from kedro.pipeline import Node, Pipeline

from .nodes import (
    add_identity_missingness_features,
    add_transaction_amount_features,
    add_transaction_time_features,
    merge_transaction_identity,
    reduce_memory,
    select_identity_features,
    select_vesta_features,
)
from .feature_selection import select_supervised_features

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=merge_transaction_identity,
                inputs=["train_transaction", "train_identity"],
                outputs="merged_data",
                name="merge_transaction_identity_node",
            ),
            Node(
                func=reduce_memory,
                inputs="merged_data",
                outputs="optimized_train",
                name="reduce_memory_node",
            ),
            Node(
                func=select_supervised_features,
                inputs=["optimized_train", "params:model_options", "params:feature_selection_options"],
                outputs=["selected_v_features", "selected_id_features", "feature_selection_report"],
                name="select_supervised_features_node",
            ),
            Node(
                func=select_vesta_features,
                inputs=["optimized_train", "selected_v_features"],
                outputs="vesta_selected_train",
                name="select_vesta_features_node",
            ),
            Node(
                func=add_identity_missingness_features,
                inputs="vesta_selected_train",
                outputs="id_missingness_featured_train",
                name="add_identity_missingness_features_node",
            ),
            Node(
                func=select_identity_features,
                inputs=["id_missingness_featured_train", "selected_id_features"],
                outputs="selected_feature_train",
                name="select_identity_features_node",
            ),
            Node(
                func=add_transaction_time_features,
                inputs="selected_feature_train",
                outputs="time_featured_train",
                name="add_transaction_time_features_node",
            ),
            Node(
                func=add_transaction_amount_features,
                inputs="time_featured_train",
                outputs="feature_engineered_train",
                name="add_transaction_amount_features_node",
            ),
        ]
    )
