"""Train Ridge (and optional challengers) on log(price) with honest preprocessing and metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from housingprices.challengers import (
    build_elasticnet_pipeline,
    build_hgbr_pipeline,
    residual_sigma_log,
)
from housingprices.preprocess import HousingPreprocessor, log_price_targets

DEFAULT_ALPHAS = np.logspace(-2, 4, 60).tolist()
RANDOM_STATE = 0
LOG_INTERVAL_Z = 1.96


def _test_metrics_pack(y_test_log: np.ndarray, y_hat_log: np.ndarray) -> dict[str, float]:
    y_test_dollar = np.exp(y_test_log)
    y_hat_dollar = np.exp(y_hat_log)
    return {
        "test_r2_log": float(r2_score(y_test_log, y_hat_log)),
        "test_mae_log": float(mean_absolute_error(y_test_log, y_hat_log)),
        "test_rmse_log": float(np.sqrt(mean_squared_error(y_test_log, y_hat_log))),
        "test_mae_dollar": float(mean_absolute_error(y_test_dollar, y_hat_dollar)),
        "test_rmse_dollar": float(np.sqrt(mean_squared_error(y_test_dollar, y_hat_dollar))),
        "test_medae_dollar": float(np.median(np.abs(y_test_dollar - y_hat_dollar))),
        "test_mape_pct": float(np.mean(np.abs(y_test_dollar - y_hat_dollar) / y_test_dollar) * 100),
    }


def _median_baseline_mae_dollar(y_true_log: np.ndarray, X_train_df: pd.DataFrame, X_test_df: pd.DataFrame) -> float:
    med = float(np.median(np.exp(log_price_targets(X_train_df))))
    actual = np.exp(y_true_log)
    pred = np.full_like(actual, med)
    return float(mean_absolute_error(actual, pred))


def _beds_baths_median_baseline_mae(
    y_true_log: np.ndarray, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> float:
    train_df = train_df.copy()
    train_df["_pb"] = list(zip(train_df["beds"], train_df["baths"]))
    med_map = train_df.groupby("_pb")["price"].median()
    global_med = float(train_df["price"].median())
    actual = np.exp(y_true_log)
    preds = []
    for _, row in test_df.iterrows():
        key = (row["beds"], row["baths"])
        preds.append(float(med_map.get(key, global_med)))
    return float(mean_absolute_error(actual, np.array(preds)))


def _try_chronological_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    min_rows: int = 24,
    min_parse_fraction: float = 0.45,
) -> tuple[Optional[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]], str]:
    """
    When ``sale_date`` parses for enough rows, return chronological train/test.
    Otherwise return (None, reason) so callers can fall back to random split.
    """
    if "sale_date" not in df.columns:
        return None, "no_sale_date_column"
    work = df.copy()
    work["_sd"] = pd.to_datetime(work["sale_date"], errors="coerce")
    ok = work["_sd"].notna()
    n_ok = int(ok.sum())
    n = len(work)
    if n_ok < min_rows:
        return None, f"too_few_parsed_dates_{n_ok}_min_{min_rows}"
    if n_ok / max(n, 1) < min_parse_fraction:
        return None, f"low_parse_rate_{n_ok / max(n, 1):.3f}_min_{min_parse_fraction}"
    work = work.loc[ok].sort_values("_sd")
    n = len(work)
    n_test = max(1, int(round(n * test_size)))
    if n - n_test < 10:
        return None, "train_too_small_after_time_split"
    train_df = work.iloc[: n - n_test].drop(columns=["_sd"])
    test_df = work.iloc[n - n_test :].drop(columns=["_sd"])
    info = {
        "train_sale_date_min": str(work["_sd"].iloc[0].date()),
        "train_sale_date_max": str(work["_sd"].iloc[n - n_test - 1].date()),
        "test_sale_date_min": str(work["_sd"].iloc[n - n_test].date()),
        "test_sale_date_max": str(work["_sd"].iloc[-1].date()),
        "rows_dropped_missing_sale_date": int((~ok).sum()),
        "rows_used_in_time_split": n,
        "min_parse_fraction": min_parse_fraction,
    }
    return (train_df, test_df, info), "ok"


def train_pipeline(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    winsor_quantile: float = 0.95,
    alphas: Optional[list[float]] = None,
    force_random_split: bool = False,
    compare_challengers: bool = False,
    promote_challenger: bool = False,
    min_time_split_parse_fraction: float = 0.45,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Hold out test data: **chronological by ``sale_date`` by default** when the column
    exists and enough values parse; otherwise random split. Winsor threshold from train only.

    If ``compare_challengers``, also fit ElasticNet and HistGradientBoosting on the same
    training rows. If ``promote_challenger`` and the split was chronological, the saved
    primary pipeline is the model with lowest ``test_mae_dollar`` among challengers that
    strictly improve over Ridge on that metric.
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    chrono_reason = "ok"
    if not force_random_split:
        chrono, chrono_reason = _try_chronological_split(
            df,
            test_size=test_size,
            min_parse_fraction=min_time_split_parse_fraction,
        )
    else:
        chrono = None
        chrono_reason = "forced_random_split"

    if chrono is not None:
        train_df, test_df, tinfo = chrono
        temporal_extra: dict[str, Any] = {"split": "chronological_by_sale_date", **tinfo}
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        temporal_extra = {
            "split": "random_holdout",
            "random_state": random_state,
            "chronological_not_used_reason": chrono_reason,
        }

    n_train_raw = len(train_df)
    thr = float(train_df["price"].astype(float).quantile(winsor_quantile))
    train_fit = train_df[train_df["price"].astype(float) < thr].copy()
    n_train_w = len(train_fit)
    if n_train_w < 10:
        raise ValueError("Too few training rows after winsorization; check data or quantile.")

    y_train = log_price_targets(train_fit)
    y_test_log = log_price_targets(test_df)

    pipe_ridge = Pipeline(
        steps=[
            ("prep", HousingPreprocessor()),
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.asarray(alphas))),
        ]
    )
    pipe_ridge.fit(train_fit, y_train)
    chosen_alpha = float(pipe_ridge.named_steps["ridge"].alpha_)
    y_hat_ridge = pipe_ridge.predict(test_df)

    metrics_ridge = _test_metrics_pack(y_test_log, y_hat_ridge)
    metrics_ridge.update(
        {
            "baseline_mae_dollar_median_train": _median_baseline_mae_dollar(y_test_log, train_fit, test_df),
            "baseline_mae_dollar_median_by_beds_baths": _beds_baths_median_baseline_mae(
                y_test_log, train_fit, test_df
            ),
            "test_rows_with_price_above_train_winsor_threshold": int(
                (test_df["price"].astype(float) >= thr).sum()
            ),
            "train_winsor_threshold_price": thr,
        }
    )

    sigma_log = residual_sigma_log(pipe_ridge, train_fit, y_train)

    selected_name = "ridge"
    selected_pipe: Pipeline = pipe_ridge
    selected_sigma = sigma_log
    challenger_report: dict[str, Any] = {}

    if compare_challengers:
        cv = min(5, max(3, len(train_fit) // 6))
        pipe_enet = build_elasticnet_pipeline(random_state, cv=cv)
        pipe_enet.fit(train_fit, y_train)
        y_hat_enet = pipe_enet.predict(test_df)
        m_enet = _test_metrics_pack(y_test_log, y_hat_enet)

        pipe_hgb = build_hgbr_pipeline(random_state)
        pipe_hgb.fit(train_fit, y_train)
        y_hat_hgb = pipe_hgb.predict(test_df)
        m_hgb = _test_metrics_pack(y_test_log, y_hat_hgb)

        challenger_report = {
            "ridge": {"test_mae_dollar": metrics_ridge["test_mae_dollar"], "chosen_alpha": chosen_alpha},
            "elasticnet": {
                "test_mae_dollar": m_enet["test_mae_dollar"],
                "l1_ratio": float(pipe_enet.named_steps["enet"].l1_ratio_),
                "alpha": float(pipe_enet.named_steps["enet"].alpha_),
            },
            "hist_gradient_boosting": {"test_mae_dollar": m_hgb["test_mae_dollar"]},
        }

        if promote_challenger and temporal_extra.get("split") == "chronological_by_sale_date":
            r_mae = metrics_ridge["test_mae_dollar"]
            candidates = [
                ("ridge", pipe_ridge, metrics_ridge["test_mae_dollar"], sigma_log),
                (
                    "elasticnet",
                    pipe_enet,
                    m_enet["test_mae_dollar"],
                    residual_sigma_log(pipe_enet, train_fit, y_train),
                ),
                (
                    "hist_gradient_boosting",
                    pipe_hgb,
                    m_hgb["test_mae_dollar"],
                    residual_sigma_log(pipe_hgb, train_fit, y_train),
                ),
            ]
            best_name, best_pipe, best_mae, best_sig = min(candidates, key=lambda x: x[2])
            if best_name != "ridge" and best_mae < r_mae:
                selected_name = best_name
                selected_pipe = best_pipe
                selected_sigma = best_sig

    y_hat_sel = selected_pipe.predict(test_df)
    final_metrics = _test_metrics_pack(y_test_log, y_hat_sel)
    final_metrics.update(
        {
            "baseline_mae_dollar_median_train": metrics_ridge["baseline_mae_dollar_median_train"],
            "baseline_mae_dollar_median_by_beds_baths": metrics_ridge["baseline_mae_dollar_median_by_beds_baths"],
            "test_rows_with_price_above_train_winsor_threshold": metrics_ridge[
                "test_rows_with_price_above_train_winsor_threshold"
            ],
            "train_winsor_threshold_price": metrics_ridge["train_winsor_threshold_price"],
        }
    )

    meta = {
        "metrics": final_metrics,
        "primary_model": selected_name,
        "chosen_alpha": chosen_alpha if selected_name == "ridge" else None,
        "winsor_threshold_price": thr,
        "n_train_raw": n_train_raw,
        "n_train_after_winsor": n_train_w,
        "n_test": len(test_df),
        "alphas_searched": list(map(float, alphas)),
        "split": temporal_extra,
        "sigma_log_train_residual": selected_sigma,
        "interval_method": "log_symmetric_normal_approx",
        "interval_z": LOG_INTERVAL_Z,
        "interval_note": "pred_log ± z*sigma_log_train on primary model train residuals; exp for dollar bounds. Empirical coverage via ledger.",
        "challenger_comparison": challenger_report or None,
    }

    return selected_pipe, meta


def save_artifacts(
    pipe: Pipeline,
    manifest: dict[str, Any],
    *,
    model_path: Path,
    manifest_path: Path,
) -> None:
    model_path = Path(model_path)
    manifest_path = Path(manifest_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def score_dataframe(
    pipe: Pipeline,
    df: pd.DataFrame,
    *,
    sigma_log: Optional[float] = None,
    z: float = LOG_INTERVAL_Z,
) -> pd.DataFrame:
    """Add pred_log_price, pred_price (dollars), optional symmetric log intervals."""
    out = df.copy()
    pred_log = pipe.predict(out)
    out["pred_log_price"] = pred_log
    out["pred_price"] = np.exp(pred_log)
    if sigma_log is not None and sigma_log > 0:
        out["pred_log_low"] = pred_log - z * sigma_log
        out["pred_log_high"] = pred_log + z * sigma_log
        out["pred_price_low"] = np.exp(out["pred_log_low"])
        out["pred_price_high"] = np.exp(out["pred_log_high"])
    if "price" in out.columns:
        out["error_dollar"] = out["pred_price"] - out["price"].astype(float)
        out["error_pct"] = (out["error_dollar"] / out["price"].astype(float)) * 100.0
    return out
