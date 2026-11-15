"""Reproduce Colab notebooks 2 and 3 feature selection on the raw CSVs."""

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_sha256(features: list[str]) -> str:
    return hashlib.sha256(json.dumps(features, separators=(",", ":")).encode()).hexdigest()


def prepared_matrix_sha256(frame: pd.DataFrame) -> str:
    """Use the same fingerprint as notebook 3's diagnostic cell."""
    digest = hashlib.sha256()
    for col in frame.columns:
        digest.update(col.encode() + b"\0")
        series = frame[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            digest.update(b"category\0")
            categories = [str(value) for value in series.cat.categories]
            digest.update(json.dumps(categories, separators=(",", ":")).encode())
            digest.update(series.cat.codes.to_numpy(dtype="<i4").tobytes())
        else:
            digest.update(b"numeric\0")
            values = series.to_numpy(dtype=np.float32, na_value=np.nan)
            missing = np.isnan(values)
            digest.update(missing.tobytes())
            digest.update(np.where(missing, 0, values).astype("<f4").tobytes())
    return digest.hexdigest()


def select_by_gain(model: xgb.XGBClassifier, columns: pd.Index) -> list[str]:
    importance = pd.Series(model.feature_importances_, index=columns)
    ranked = importance.sort_values(ascending=False, kind="stable")
    if ranked.sum() <= 0:
        raise ValueError("No positive feature gain")
    count = int(np.searchsorted((ranked.cumsum() / ranked.sum()).to_numpy(), 0.95, side="left")) + 1
    return ranked.head(count).index.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/data/01_raw"))
    parser.add_argument("--notebook", type=Path, default=Path("/workspace/notebooks/03. IEEE-CIS Fraud Detection - ID features exploration.ipynb"))
    parser.add_argument("--group", choices=("v", "id", "both"), default="both")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--reducer", choices=("notebook", "project"), default="notebook")
    args = parser.parse_args()

    if args.reducer == "notebook":
        # Execute the notebooks' downcast function, so this uses their exact rules.
        notebook = json.loads(args.notebook.read_text())
        namespace = {"pd": pd, "np": np}
        exec("".join(notebook["cells"][5]["source"]), namespace)
        reduce_memory = namespace["reduce_memory"]
    else:
        from ieee_cis_fraud_detection.pipelines.data_processing.nodes import reduce_memory

    transaction_path = args.data_dir / "train_transaction.csv"
    identity_path = args.data_dir / "train_identity.csv"
    result = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "xgboost_build": xgb.build_info(),
            "n_jobs": args.n_jobs,
        },
        "input_sha256": {
            "train_transaction.csv": file_sha256(transaction_path),
            "train_identity.csv": file_sha256(identity_path),
        },
    }

    groups = ("v", "id") if args.group == "both" else (args.group,)
    for group in groups:
        # Read only the group being ranked. The values and row order are the
        # same as the wide merge, while peak RAM stays below Docker Desktop's
        # allocation on a 16 GB Mac.
        if group == "v":
            header = pd.read_csv(transaction_path, nrows=0).columns
            usecols = ["TransactionDT", "isFraud"] + [col for col in header if col.startswith("V")]
            merged = reduce_memory(pd.read_csv(transaction_path, usecols=usecols), verbose=False)
        else:
            transaction = reduce_memory(
                pd.read_csv(transaction_path, usecols=["TransactionID", "TransactionDT", "isFraud"]),
                verbose=False,
            )
            identity = reduce_memory(pd.read_csv(identity_path), verbose=False)
            merged = reduce_memory(pd.merge(transaction, identity, on="TransactionID", how="left"), verbose=False)
            del transaction, identity

        order = merged["TransactionDT"].sort_values(kind="stable").index
        train_idx = order[: int(len(order) * 0.8)]
        holdout_idx = order[int(len(order) * 0.8) :]
        y = merged.loc[train_idx, "isFraud"]
        weight = (len(y) - y.sum()) / y.sum()
        split = {
            "selection_rows": len(train_idx),
            "holdout_rows": len(holdout_idx),
            "selection_time_max": int(merged.loc[train_idx, "TransactionDT"].max()),
            "holdout_time_min": int(merged.loc[holdout_idx, "TransactionDT"].min()),
        }
        if "split" in result and result["split"] != split:
            raise ValueError("V and ID selectors used different time boundaries")
        result["split"] = split
        common = dict(
            objective="binary:logistic",
            eval_metric="aucpr",
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=args.n_jobs,
            random_state=42,
            scale_pos_weight=weight,
            importance_type="gain",
            tree_method="hist",
        )

        if group == "v":
            v_columns = [col for col in merged if col.startswith("V")]
            X_v = merged.loc[train_idx, v_columns].astype("float32")
            v_ranker = xgb.XGBClassifier(n_estimators=500, **common)
            v_ranker.fit(X_v, y)
            selected = select_by_gain(v_ranker, X_v.columns)
            result["v"] = {"count": len(selected), "sha256": list_sha256(selected), "features": selected}
            del X_v, v_ranker
        else:
            id_columns = [col for col in merged if col.startswith("id_")]
            id_data = merged[id_columns].copy()
            for col in id_columns:
                if id_data.loc[train_idx, col].isna().any():
                    id_data[f"{col}_is_missing"] = id_data[col].isna().astype("int8")
            X_id = id_data.loc[train_idx].copy()
            for col in id_columns:
                if (
                    pd.api.types.is_object_dtype(X_id[col])
                    or pd.api.types.is_string_dtype(X_id[col])
                    or isinstance(X_id[col].dtype, pd.CategoricalDtype)
                ):
                    X_id[col] = X_id[col].astype("object").fillna("__MISSING__").astype("category")
            id_ranker = xgb.XGBClassifier(n_estimators=300, enable_categorical=True, **common)
            id_ranker.fit(X_id, y)
            selected = select_by_gain(id_ranker, X_id.columns)
            result["id"] = {
                "input_matrix_sha256": prepared_matrix_sha256(X_id),
                "count": len(selected),
                "sha256": list_sha256(selected),
                "features": selected,
            }
            del X_id, id_data, id_ranker

        del merged, y, order, train_idx, holdout_idx
        gc.collect()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
