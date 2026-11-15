"""Convert the narrow Linux selector result into Kedro input artifacts."""

import json
from pathlib import Path


data_root = Path("/app/data")
result = json.loads((data_root / "08_reporting/docker_selection_result.json").read_text())
primary = data_root / "03_primary"
primary.mkdir(parents=True, exist_ok=True)

for group, name in [("v", "selected_v_features"), ("id", "selected_id_features")]:
    (primary / f"{name}.json").write_text(json.dumps(result[group]["features"], indent=2))

split = result["split"]
report = {
    "method": "XGBoost gain importance fitted on chronological training rows only, with separate narrow CSV reads",
    "environment": result["environment"],
    "input_sha256": result["input_sha256"],
    "xgboost_version": result["environment"]["xgboost"],
    "pandas_version": result["environment"]["pandas"],
    "v_feature_sha256": result["v"]["sha256"],
    "id_feature_sha256": result["id"]["sha256"],
    "split_strategy": "time_ordered",
    "time_column": "TransactionDT",
    "test_size": 0.2,
    "n_selection_rows": split["selection_rows"],
    "n_held_out_rows": split["holdout_rows"],
    "selection_time_max": split["selection_time_max"],
    "holdout_time_min": split["holdout_time_min"],
    "n_selected_v_features": result["v"]["count"],
    "n_selected_id_features": result["id"]["count"],
    "selected_v_features": result["v"]["features"],
    "selected_id_features": result["id"]["features"],
}
report_path = data_root / "08_reporting/feature_selection_report.json"
report_path.write_text(json.dumps(report, indent=2))
print("Prepared Kedro feature artifacts:", result["v"]["count"], "V and", result["id"]["count"], "ID")
