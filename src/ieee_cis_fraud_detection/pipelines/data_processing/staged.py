"""Memory-conscious feature table construction for the Linux Docker run."""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from kedro.pipeline import Node, Pipeline

from .nodes import reduce_memory


def load_selected_feature_table(
    raw_paths: dict,
    selected_v_features: list[str],
    selected_id_features: list[str],
) -> pd.DataFrame:
    """Read only columns needed by the model, following notebook 4's load path."""
    transaction_path = Path(raw_paths["transaction"])
    identity_path = Path(raw_paths["identity"])
    selected_v = set(selected_v_features)
    selected_id_sources = {name.removesuffix("_is_missing") for name in selected_id_features}

    transaction = reduce_memory(pd.read_csv(
        transaction_path,
        usecols=lambda col: not col.startswith("V") or col in selected_v,
    ), verbose=False)
    identity = reduce_memory(pd.read_csv(
        identity_path,
        usecols=lambda col: not col.startswith("id_") or col in selected_id_sources,
    ), verbose=False)
    table = pd.merge(transaction, identity, on="TransactionID", how="left")
    del transaction, identity
    gc.collect()

    for col in selected_id_features:
        if col.endswith("_is_missing"):
            source = col.removesuffix("_is_missing")
            table[col] = table[source].isna().astype("int8")
    unused_id_sources = [
        col for col in table if col.startswith("id_") and col not in selected_id_features
    ]
    table.drop(columns=unused_id_sources, inplace=True)
    missing = (set(selected_v_features) | set(selected_id_features)) - set(table.columns)
    if missing:
        raise ValueError(f"Selected features missing from staged table: {sorted(missing)}")

    table["Transaction_day"] = table["TransactionDT"] // (60 * 60 * 24)
    table["Transaction_hour"] = (table["TransactionDT"] // (60 * 60)) % 24
    table["TransactionAmt_log1p"] = np.log1p(table["TransactionAmt"])
    return table


def create_pipeline() -> Pipeline:
    return Pipeline([
        Node(
            func=load_selected_feature_table,
            inputs=["params:raw_data_paths", "selected_v_features", "selected_id_features"],
            outputs="feature_engineered_train",
            name="load_staged_selected_feature_table_node",
        ),
    ])
