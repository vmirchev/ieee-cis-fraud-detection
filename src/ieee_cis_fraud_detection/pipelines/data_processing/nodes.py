import pandas as pd
import numpy as np
import warnings
from contextlib import nullcontext

def merge_transaction_identity(train_transaction, train_identity):
    return pd.merge(train_transaction, train_identity, on='TransactionID', how='left')


def reduce_memory(df: pd.DataFrame, verbose: bool = True, ignore_warnings: bool = False, inplace: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to reduce DataFrame memory usage.
    """
    if not inplace:
        df = df.copy()

    start_mem = df.memory_usage(deep=True).sum() / 1e6

    warning_context = warnings.catch_warnings() if ignore_warnings else nullcontext()

    with warning_context:
        if ignore_warnings:
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="overflow encountered in cast",
            )

        for col in df.columns:
            series = df[col]

            if pd.api.types.is_integer_dtype(series):
                c_min = series.min()
                c_max = series.max()

                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = series.astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = series.astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = series.astype(np.int32)
                else:
                    df[col] = series.astype(np.int64)

            elif pd.api.types.is_float_dtype(series):
                c_min = series.min()
                c_max = series.max()

                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = series.astype(np.float32)
                else:
                    df[col] = series.astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1e6

    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem else 0
        print(f"Memory: {start_mem:.2f} MB → {end_mem:.2f} MB ({reduction:.1f}% reduction)")

    return df

def add_transaction_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TransactionDT" in df.columns:
        df["Transaction_day"] = df["TransactionDT"] // (60 * 60 * 24)
        df["Transaction_hour"] = (df["TransactionDT"] // (60 * 60)) % 24

    return df


def add_transaction_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"])
    return df


def select_vesta_features(
    df: pd.DataFrame,
    selected_v_features: list[str],
) -> pd.DataFrame:
    selected_v_features = set(selected_v_features)

    keep_columns = [
        col
        for col in df.columns
        if not col.startswith("V") or col in selected_v_features
    ]

    return df[keep_columns]


def add_identity_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    id_columns = [col for col in df.columns if col.startswith("id_")]

    for col in id_columns:
        df[f"{col}_is_missing"] = df[col].isna().astype("int8")

    return df


def select_identity_features(
    df: pd.DataFrame,
    selected_id_features: list[str],
) -> pd.DataFrame:
    selected_id_features = set(selected_id_features)

    keep_columns = [
        col
        for col in df.columns
        if not col.startswith("id_") or col in selected_id_features
    ]

    return df[keep_columns]
