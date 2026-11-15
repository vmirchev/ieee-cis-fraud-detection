import numpy as np
import pandas as pd

from ieee_cis_fraud_detection.pipelines.data_processing.feature_selection import select_supervised_features
from ieee_cis_fraud_detection.validation import time_ordered_indices


def test_selection_never_uses_holdout_features_or_labels():
    rng = np.random.default_rng(42)
    data = pd.DataFrame({
        "TransactionDT": np.arange(40),
        "isFraud": [0, 1] * 20,
        "V1": rng.normal(size=40),
        "V2": rng.normal(size=40),
        "V3": rng.normal(size=40),
        "id_01": rng.normal(size=40),
        "id_02": ["A", "B"] * 20,
    })
    data.loc[0, "id_01"] = np.nan
    split = {"split_strategy": "time_ordered", "time_column": "TransactionDT", "test_size": 0.2, "target": "isFraud"}
    options = {"v_gain_fraction": 0.95, "id_gain_fraction": 0.95, "v_n_estimators": 5, "id_n_estimators": 5,
               "learning_rate": 0.05, "max_depth": 2, "subsample": 0.8,
               "colsample_bytree": 0.8, "n_jobs": 1, "random_state": 42}

    original = select_supervised_features(data, split, options)
    changed = data.copy()
    _, holdout_idx = time_ordered_indices(data, "TransactionDT", 0.2)
    changed.loc[holdout_idx, "isFraud"] = 1 - changed.loc[holdout_idx, "isFraud"]
    changed.loc[holdout_idx, ["V1", "V2", "V3", "id_01"]] = 9999
    changed.loc[holdout_idx, "id_02"] = "new_holdout_category"
    altered = select_supervised_features(changed, split, options)

    assert original[0] == altered[0]
    assert original[1] == altered[1]
    assert original[2]["n_selection_rows"] == 32
    assert original[2]["n_held_out_rows"] == 8
    assert original[2]["selection_time_max"] < original[2]["holdout_time_min"]
