"""Optional ElasticNet and HistGradientBoosting baselines (same preprocessing contract as Ridge)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from housingprices.preprocess import HousingPreprocessor


def build_elasticnet_pipeline(random_state: int, cv: int = 5) -> Pipeline:
    return Pipeline(
        [
            ("prep", HousingPreprocessor()),
            ("scale", StandardScaler()),
            (
                "enet",
                ElasticNetCV(
                    l1_ratio=(0.15, 0.5, 0.85),
                    eps=1e-3,
                    n_alphas=24,
                    cv=cv,
                    random_state=random_state,
                    max_iter=8000,
                ),
            ),
        ]
    )


def build_hgbr_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("prep", HousingPreprocessor()),
            (
                "hgb",
                HistGradientBoostingRegressor(
                    max_depth=3,
                    max_iter=120,
                    learning_rate=0.08,
                    min_samples_leaf=5,
                    l2_regularization=0.2,
                    random_state=random_state,
                ),
            ),
        ]
    )


def residual_sigma_log(pipe: Pipeline, X, y_log: np.ndarray) -> float:
    pred = pipe.predict(X)
    r = y_log - pred
    if len(r) < 2:
        return float(np.std(r))
    return float(np.std(r, ddof=1))
