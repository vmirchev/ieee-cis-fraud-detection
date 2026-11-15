import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_frequency=0.01, missing_value="Missing", rare_value="Rare"):
        self.min_frequency = min_frequency
        self.missing_value = missing_value
        self.rare_value = rare_value

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = X.columns.to_list()
        self.frequent_categories_ = {}

        for column in X.columns:
            values = X[column].fillna(self.missing_value).astype(str)
            frequencies = values.value_counts(normalize=True)
            self.frequent_categories_[column] = set(
                frequencies[frequencies >= self.min_frequency].index
            )

        return self

    def transform(self, X):
        check_is_fitted(self, "frequent_categories_")

        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()

        for column in X.columns:
            values = X[column].fillna(self.missing_value).astype(str)
            keep = self.frequent_categories_[column]
            X[column] = values.where(values.isin(keep), self.rare_value)

        return X