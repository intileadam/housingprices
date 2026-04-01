"""Feature engineering aligned with v7 notebook: sqrt(sqft), get_dummies, fixed column contract."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


CAT_COLS = [
    "view_yn",
    "cooling_yn",
    "senior_community_yn",
    "style",
    "county",
    "new_construction_yn",
    "has_hoa",
]

# Dropped from X after dummies (matches archive/notebooks/v7_mlr_home_logprice_ridge.ipynb).
DROP_FROM_X = [
    "Address",
    "link",
    "price",
    "log_price",
    "baths",
    "new_construction_yn_Yes",
    "view_yn_Yes",
]

# Chronological split and ledger only; stripped before feature matrix.
AUX_COLUMNS = ["sale_date"]

# If present in the DataFrame, coerced to float and used as numeric features (0 if missing).
OPTIONAL_NUMERIC = ["days_on_market", "latitude", "longitude"]


class HousingPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transform cleaned listing CSV rows into the numeric matrix used by Ridge on log(price).

    ``fit`` learns the dummy column set from training data; ``transform`` reindexes so
    missing dummies are zero (stable scoring when new categories appear).
    """

    def __init__(self, cat_cols: Optional[list[str]] = None, drop_from_x: Optional[list[str]] = None):
        self.cat_cols = cat_cols if cat_cols is not None else list(CAT_COLS)
        self.drop_from_x = drop_from_x if drop_from_x is not None else list(DROP_FROM_X)
        self.feature_columns_: Optional[list[str]] = None

    def fit(self, X, y=None):
        X_df = self._as_df(X)
        engineered = self._engineer(X_df, align_to=None)
        self.feature_columns_ = engineered.columns.tolist()
        return self

    def transform(self, X):
        if self.feature_columns_ is None:
            raise RuntimeError("Call fit before transform.")
        X_df = self._as_df(X)
        engineered = self._engineer(X_df, align_to=self.feature_columns_)
        return engineered.values.astype(np.float64)

    def get_feature_names_out(self, input_features=None):
        if self.feature_columns_ is None:
            raise RuntimeError("Call fit before get_feature_names_out.")
        return np.array(self.feature_columns_, dtype=object)

    @staticmethod
    def _as_df(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        raise TypeError("HousingPreprocessor expects a pandas DataFrame.")

    def _engineer(self, X: pd.DataFrame, align_to: Optional[list[str]]) -> pd.DataFrame:
        out = X.copy()
        for c in AUX_COLUMNS:
            if c in out.columns:
                out = out.drop(columns=[c])
        if "lot_size_sqft" in out.columns:
            digits = out["lot_size_sqft"].astype(str).str.replace(r"[^\d]", "", regex=True)
            out["lot_sqft"] = pd.to_numeric(digits, errors="coerce").fillna(0.0)
            out = out.drop(columns=["lot_size_sqft"])
        for c in OPTIONAL_NUMERIC:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        if "sqft" not in out.columns:
            raise ValueError("Expected column 'sqft'.")
        out["sqft"] = np.sqrt(out["sqft"].astype(float))
        out = pd.get_dummies(out, columns=self.cat_cols, drop_first=True)
        to_drop = [c for c in self.drop_from_x if c in out.columns]
        out = out.drop(columns=to_drop, axis=1)
        if align_to is not None:
            out = out.reindex(columns=align_to, fill_value=0.0)
        return out


def log_price_targets(df: pd.DataFrame) -> np.ndarray:
    """Natural log of ``price`` for use as regression target."""
    return np.log(df["price"].astype(float).values)
