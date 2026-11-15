"""Shared holdout boundary for feature selection and model evaluation."""

import pandas as pd


def time_ordered_indices(data: pd.DataFrame, time_column: str, test_size: float):
    """Return deterministic earlier/later row indices for a time holdout."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if time_column not in data.columns:
        raise KeyError(time_column)
    split_idx = int(len(data) * (1 - test_size))
    ordered = data[time_column].sort_values(kind="stable").index
    return ordered[:split_idx], ordered[split_idx:]
