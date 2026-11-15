import pandas as pd
import pytest
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from sklearn.linear_model import LogisticRegression

from ieee_cis_fraud_detection.pipelines.data_processing.nodes import add_identity_missingness_features

from ieee_cis_fraud_detection.pipelines.data_science import (
    create_pipeline as create_ds_pipeline,
)
from ieee_cis_fraud_detection.pipelines.data_science.nodes import (
    compare_models,
    compare_models_without_baseline,
    create_feature_groups,
    create_numeric_feature_list,
    evaluate_selected_threshold,
    evaluate_model,
    select_final_model,
    select_final_threshold,
    split_data,
    train_model,
    tune_model_threshold,
    tune_threshold,
)


@pytest.fixture
def fraud_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": range(20),
            "TransactionDT": range(1000, 1020),
            "TransactionAmt": [
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                80,
                82,
                84,
                86,
                88,
                90,
                92,
                94,
                96,
                98,
            ],
            "Transaction_day": [0] * 10 + [1] * 10,
            "Transaction_hour": list(range(10)) * 2,
            "TransactionAmt_log1p": [
                2.39,
                2.56,
                2.71,
                2.83,
                2.94,
                3.04,
                3.14,
                3.22,
                3.30,
                3.37,
                4.39,
                4.42,
                4.44,
                4.47,
                4.49,
                4.51,
                4.53,
                4.55,
                4.57,
                4.58,
            ],
            "ProductCD": ["W", "C"] * 10,
            "isFraud": [0, 1] * 10,
        }
    )


@pytest.fixture
def model_options() -> dict:
    return {
        "target": "isFraud",
        "drop_columns": ["TransactionID", "TransactionDT"],
        "split_strategy": "random_stratified",
        "time_column": "TransactionDT",
        "test_size": 0.25,
        "random_state": 42,
        "stratify": True,
    }


@pytest.fixture
def baseline_model_options() -> dict:
    return {
        "model_name": "LogisticRegression",
        "random_state": 42,
        "class_weight": "balanced",
        "max_iter": 100,
        "solver": "lbfgs",
        "rare_min_frequency": 0.005,
    }


@pytest.fixture
def feature_grouping_options() -> dict:
    return {
        "id_like_numeric": [],
        "known_categorical": ["ProductCD"],
        "exclude_from_features": ["TransactionID", "TransactionDT", "isFraud"],
    }


@pytest.fixture
def threshold_options() -> dict:
    return {
        "target": "isFraud",
        "thresholds": [0.25, 0.5, 0.75],
    }


def test_split_data_uses_fraud_target_and_drops_id_columns(
    fraud_data: pd.DataFrame, model_options: dict
):
    X_train, X_test, y_train, y_test = split_data(fraud_data, model_options)

    assert len(X_train) == 15
    assert len(X_test) == 5
    assert y_train.columns.tolist() == ["isFraud"]
    assert y_test.columns.tolist() == ["isFraud"]
    assert "TransactionID" not in X_train.columns
    assert "TransactionDT" not in X_train.columns
    assert "isFraud" not in X_train.columns


def test_split_data_supports_time_ordered_validation(
    fraud_data: pd.DataFrame, model_options: dict
):
    model_options = {
        **model_options,
        "split_strategy": "time_ordered",
        "test_size": 0.25,
    }

    X_train, X_test, y_train, y_test = split_data(fraud_data, model_options)

    assert X_train.index.tolist() == list(range(15))
    assert X_test.index.tolist() == list(range(15, 20))
    assert y_train.index.tolist() == list(range(15))
    assert y_test.index.tolist() == list(range(15, 20))
    assert "TransactionID" not in X_train.columns
    assert "TransactionDT" not in X_train.columns


def test_split_data_missing_target_raises_key_error(
    fraud_data: pd.DataFrame, model_options: dict
):
    with pytest.raises(KeyError, match="isFraud"):
        split_data(fraud_data.drop(columns="isFraud"), model_options)


def test_create_numeric_feature_list_excludes_categorical_columns(
    fraud_data: pd.DataFrame, model_options: dict
):
    X_train, *_ = split_data(fraud_data, model_options)

    model_features = create_numeric_feature_list(X_train)

    assert "TransactionAmt" in model_features
    assert "TransactionAmt_log1p" in model_features
    assert "ProductCD" not in model_features


def test_evaluate_model_from_prediction_artifact():
    predictions = pd.DataFrame(
        {
            "isFraud": [0, 0, 1, 1],
            "y_score": [0.1, 0.4, 0.6, 0.9],
        }
    )

    metrics = evaluate_model(predictions)

    assert metrics == {"roc_auc": 1.0, "pr_auc": 1.0}


def test_comparison_selects_best_pr_auc_on_shared_split(model_options: dict):
    baseline = {"roc_auc": 0.91, "pr_auc": 0.20}
    tree = {"roc_auc": 0.89, "pr_auc": 0.30}
    lightgbm = {"roc_auc": 0.93, "pr_auc": 0.40}
    xgboost = {"roc_auc": 0.92, "pr_auc": 0.35}

    split_options = {**model_options, "split_strategy": "time_ordered"}
    split_summary = {"n_train_rows": 15, "n_test_rows": 5}
    selection_report = {"split_strategy": "time_ordered", "time_column": "TransactionDT", "test_size": 0.25,
                        "n_selection_rows": 15, "n_held_out_rows": 5}
    comparison = compare_models(baseline, tree, lightgbm, xgboost, split_options, split_summary, selection_report)

    assert comparison["selection_metric"] == "pr_auc"
    assert comparison["split_options"] == split_options
    assert comparison["feature_selection"] == selection_report
    assert comparison["best_by_pr_auc"] == "lightgbm"
    assert select_final_model("baseline model", "tree model", "lightgbm model", "xgboost model", comparison) == "lightgbm model"

    with pytest.raises(ValueError, match="boundary"):
        compare_models(baseline, tree, lightgbm, xgboost, split_options, split_summary,
                       {**selection_report, "n_held_out_rows": 4})


def test_docker_comparison_skips_linear_baseline(model_options: dict):
    split_options = {**model_options, "split_strategy": "time_ordered"}
    report = compare_models_without_baseline(
        {"roc_auc": 0.8, "pr_auc": 0.3},
        {"roc_auc": 0.9, "pr_auc": 0.5},
        {"roc_auc": 0.91, "pr_auc": 0.49},
        split_options,
        {"n_train_rows": 15, "n_test_rows": 5},
        {"split_strategy": "time_ordered", "time_column": "TransactionDT",
         "test_size": 0.25, "n_selection_rows": 15, "n_held_out_rows": 5},
    )
    assert set(report["models"]) == {"tree", "lightgbm", "xgboost"}
    assert report["best_by_pr_auc"] == "lightgbm"


def test_tune_threshold_returns_best_threshold_by_f1(threshold_options: dict):
    predictions = pd.DataFrame(
        {
            "isFraud": [0, 0, 1, 1],
            "y_score": [0.1, 0.4, 0.6, 0.9],
        }
    )

    report = tune_threshold(predictions, threshold_options)

    assert len(report["thresholds"]) == 3
    assert report["best_threshold_by_f1"]["threshold"] == 0.5
    assert report["best_threshold_by_f1"]["f1"] == 1.0


def test_selected_id_missingness_is_numeric():
    data = add_identity_missingness_features(pd.DataFrame({
        "id_01": [1.0, None, 3.0], "TransactionAmt": [10.0, 20.0, 30.0],
    }))
    numeric, categorical = create_feature_groups(data, {
        "id_like_numeric": [], "known_categorical": [], "exclude_from_features": [],
    })
    assert str(data["id_01_is_missing"].dtype) == "int8"
    assert "id_01_is_missing" in numeric
    assert "id_01_is_missing" not in categorical


def test_baseline_imputer_preserves_missingness_indicator(baseline_model_options: dict):
    X = pd.DataFrame({
        "amount": [1.0, None, 2.0, 3.0, None, 4.0, 5.0, 6.0],
        "product": ["A", "B"] * 4,
    })
    y = pd.DataFrame({"isFraud": [0, 1] * 4})
    model = train_model(X, y, ["amount"], ["product"], baseline_model_options)
    imputer = model.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"]
    assert imputer.add_indicator
    assert imputer.indicator_.features_.tolist() == [0]


def test_threshold_is_chosen_on_inner_period_only(fraud_data: pd.DataFrame):
    split_options = {
        "target": "isFraud", "drop_columns": ["TransactionID", "TransactionDT"],
        "split_strategy": "time_ordered", "time_column": "TransactionDT",
        "test_size": 0.2, "random_state": 42,
    }
    options = {
        "target": "isFraud", "inner_validation_fraction": 0.25,
        "thresholds": [0.25, 0.5, 0.75],
    }
    X_train, X_holdout, y_train, y_holdout = split_data(fraud_data, split_options)
    model = LogisticRegression(random_state=42)
    report = tune_model_threshold(
        model, X_train, y_train, ["TransactionAmt"], split_options, options,
    )
    expected_model = LogisticRegression(random_state=42).fit(
        X_train.iloc[:12][["TransactionAmt"]], y_train.iloc[:12, 0],
    )
    inner_predictions = pd.DataFrame({
        "isFraud": y_train.iloc[12:, 0],
        "y_score": expected_model.predict_proba(X_train.iloc[12:][["TransactionAmt"]])[:, 1],
    })
    expected = tune_threshold(inner_predictions, options)
    assert report["best_threshold_by_f1"] == expected["best_threshold_by_f1"]
    assert report["n_inner_fit_rows"] == 12
    assert report["n_inner_validation_rows"] == 4

    holdout_predictions = pd.DataFrame({
        "isFraud": y_holdout["isFraud"], "y_score": [0.2, 0.4, 0.6, 0.8],
    })
    evaluated = evaluate_selected_threshold(report, holdout_predictions, options)
    changed = holdout_predictions.copy()
    changed["isFraud"] = 1 - changed["isFraud"]
    evaluated_changed = evaluate_selected_threshold(report, changed, options)
    assert evaluated["best_threshold_by_f1"] == evaluated_changed["best_threshold_by_f1"]
    assert evaluated["holdout_at_selected_threshold"] != evaluated_changed["holdout_at_selected_threshold"]


def test_final_threshold_follows_selected_model():
    reports = []
    for threshold in [0.2, 0.3, 0.79, 0.8]:
        reports.append({
            "selection_period": "inner_training_validation",
            "best_threshold_by_f1": {"threshold": threshold, "f1": 0.5},
            "holdout_at_selected_threshold": {"threshold": threshold, "f1": 0.4},
        })
    selected = select_final_threshold(
        {"best_by_pr_auc": "lightgbm", "selection_metric": "pr_auc"}, *reports,
    )
    assert selected["selected_model"] == "lightgbm"
    assert selected["selected_threshold"] == 0.79
    assert selected["threshold_selection_period"] == "inner_training_validation"


def test_data_science_pipeline_runs_with_synthetic_fraud_data(
    fraud_data: pd.DataFrame,
    model_options: dict,
    baseline_model_options: dict,
    feature_grouping_options: dict,
    threshold_options: dict,
):
    pipeline = create_ds_pipeline()
    catalog = DataCatalog()
    catalog["feature_engineered_train"] = fraud_data
    catalog["params:model_options"] = model_options
    catalog["params:baseline_model_options"] = baseline_model_options
    catalog["params:feature_grouping"] = feature_grouping_options
    catalog["params:threshold_options"] = threshold_options

    outputs = SequentialRunner().run(pipeline, catalog)
    metadata = outputs["baseline_model_metadata"].load()
    threshold_report = outputs["baseline_threshold_report"].load()

    assert "baseline_threshold_report" in outputs
    assert "baseline_model_metadata" in outputs
    assert metadata["target"] == "isFraud"
    assert metadata["n_features"] == 5
    assert "ProductCD" in metadata["model_features"]
    assert metadata["metrics"].keys() == {"roc_auc", "pr_auc"}
    assert threshold_report["best_threshold_by_f1"]["threshold"] in {
        0.25,
        0.5,
        0.75,
    }
