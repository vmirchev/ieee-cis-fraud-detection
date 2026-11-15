import pandas as pd
import pytest

from ieee_cis_fraud_detection.pipelines.data_science.boosted_nodes import (
    create_boosted_model_metadata,
    train_lightgbm_model,
    train_xgboost_model,
)


@pytest.fixture
def training_data():
    X = pd.DataFrame({
        "amount": [float(i) for i in range(20)],
        "distance": [float(i % 5) for i in range(20)],
        "product": ["A", "B"] * 10,
    })
    X.loc[0, "amount"] = float("nan")
    X.loc[1, "product"] = None
    y = pd.DataFrame({"isFraud": [0] * 15 + [1] * 5})
    return X, y


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost"])
def test_boosted_models_use_notebook_preprocessing_and_imbalance(training_data, model_name):
    X, y = training_data
    common = {"model_name": model_name, "rare_min_frequency": 0.005, "random_state": 42, "n_jobs": 1}
    if model_name == "lightgbm":
        parameters = {**common, "objective": "binary", "metric": "average_precision", "n_estimators": 5,
                      "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "class_weight": "balanced"}
        model = train_lightgbm_model(X, y, ["amount", "distance"], ["product"], parameters)
        assert model.named_steps["model"].class_weight == "balanced"
    else:
        parameters = {**common, "objective": "binary:logistic", "eval_metric": "aucpr", "n_estimators": 5,
                      "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
                      "min_child_weight": 1, "reg_lambda": 1.0, "tree_method": "hist"}
        model = train_xgboost_model(X, y, ["amount", "distance"], ["product"], parameters)
        assert model.named_steps["model"].scale_pos_weight == 3.0
        assert model.named_steps["model"].tree_method == "hist"

    numeric = model.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"]
    categorical = model.named_steps["preprocess"].named_transformers_["cat"].named_steps
    assert numeric.add_indicator
    assert numeric.indicator_.features_.tolist() == [0]
    assert categorical["ordinal"].unknown_value == -1
    assert categorical["rare"].min_frequency == 0.005
    assert model.predict_proba(X).shape == (20, 2)

    metadata = create_boosted_model_metadata(X, X, y, ["amount", "distance"], ["product"], model, parameters, {"pr_auc": 0.5})
    assert metadata["categorical_features"] == ["product"]
    if model_name == "xgboost":
        assert metadata["effective_scale_pos_weight"] == 3.0


def test_xgboost_rejects_single_class(training_data):
    X, _ = training_data
    y = pd.DataFrame({"isFraud": [0] * len(X)})
    with pytest.raises(ValueError, match="both fraud and non-fraud"):
        train_xgboost_model(X, y, ["amount"], ["product"], {"rare_min_frequency": 0.005})
