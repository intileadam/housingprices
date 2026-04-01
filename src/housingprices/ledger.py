"""
Append-only prediction ledger (CSV) for later reconciliation with realized sales.

Schema (one row per property per prediction run):
  run_id, timestamp_utc, model_version, data_file_sha256, zipcode, property_key,
  address, link, pred_log_price, pred_price, pred_price_low, pred_price_high,
  interval_method, interval_z, interval_sigma_log,
  actual_sale_price, sale_date, days_on_market, above_winsor_threshold,
  error_dollar, error_pct, actual_inside_interval

Use reconcile_ledger for error aggregates and empirical interval coverage when bounds exist.
"""

from __future__ import annotations

import csv
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

LEDGER_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "model_version",
    "data_file_sha256",
    "zipcode",
    "property_key",
    "address",
    "link",
    "pred_log_price",
    "pred_price",
    "pred_price_low",
    "pred_price_high",
    "interval_method",
    "interval_z",
    "interval_sigma_log",
    "actual_sale_price",
    "sale_date",
    "days_on_market",
    "above_winsor_threshold",
    "error_dollar",
    "error_pct",
    "actual_inside_interval",
]


def make_property_key(zipcode: str, address: str) -> str:
    norm = f"{zipcode}|{str(address).strip().lower()}"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def make_run_id() -> str:
    return str(uuid.uuid4())


def append_prediction_rows(
    path: Path,
    rows: list[Mapping[str, Any]],
    *,
    create: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = create and (not path.exists() or path.stat().st_size == 0)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LEDGER_COLUMNS})


def build_rows_from_scored_df(
    scored: pd.DataFrame,
    *,
    run_id: str,
    model_version: str,
    data_file_sha256: str,
    zipcode: str,
    winsor_threshold: Optional[float] = None,
    interval_method: str = "",
    interval_z: str = "",
    interval_sigma_log: str = "",
) -> list[dict[str, Any]]:
    ts = datetime.now(timezone.utc).isoformat()
    out: list[dict[str, Any]] = []
    for _, row in scored.iterrows():
        addr = row.get("Address", "")
        key = make_property_key(zipcode, str(addr))
        price = row.get("price")
        pred_log = float(row["pred_log_price"])
        pred_p = float(row["pred_price"])
        above = ""
        if winsor_threshold is not None and pd.notna(price):
            above = str(float(price) >= float(winsor_threshold))
        err_d = row.get("error_dollar", "")
        err_p = row.get("error_pct", "")
        if err_d == "" and pd.notna(price):
            err_d = pred_p - float(price)
            err_p = (err_d / float(price)) * 100.0
        p_lo = row["pred_price_low"] if "pred_price_low" in scored.columns else np.nan
        p_hi = row["pred_price_high"] if "pred_price_high" in scored.columns else np.nan
        p_lo_s = "" if pd.isna(p_lo) else float(p_lo)
        p_hi_s = "" if pd.isna(p_hi) else float(p_hi)
        inside = ""
        if pd.notna(price) and p_lo_s != "" and p_hi_s != "":
            lo, hi = float(p_lo_s), float(p_hi_s)
            ap = float(price)
            inside = str(lo <= ap <= hi)
        sale_dt = row.get("sale_date", "")
        if pd.isna(sale_dt):
            sale_dt = ""
        dom = row.get("days_on_market", "")
        if pd.isna(dom):
            dom = ""
        out.append(
            {
                "run_id": run_id,
                "timestamp_utc": ts,
                "model_version": model_version,
                "data_file_sha256": data_file_sha256,
                "zipcode": zipcode,
                "property_key": key,
                "address": addr,
                "link": row.get("link", ""),
                "pred_log_price": pred_log,
                "pred_price": pred_p,
                "pred_price_low": p_lo_s if p_lo_s != "" else "",
                "pred_price_high": p_hi_s if p_hi_s != "" else "",
                "interval_method": interval_method,
                "interval_z": interval_z,
                "interval_sigma_log": interval_sigma_log,
                "actual_sale_price": float(price) if pd.notna(price) else "",
                "sale_date": sale_dt,
                "days_on_market": dom,
                "above_winsor_threshold": above,
                "error_dollar": err_d if err_d != "" else "",
                "error_pct": err_p if err_p != "" else "",
                "actual_inside_interval": inside,
            }
        )
    return out


def load_ledger(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    df = pd.read_csv(path)
    for c in LEDGER_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def reconcile(path: Path) -> dict[str, float]:
    """Aggregate errors where actual_sale_price is present (e.g. after backfill)."""
    df = load_ledger(path)
    if df.empty:
        return {"n_rows": 0.0, "n_with_actual": 0.0}
    sub = df[pd.to_numeric(df["actual_sale_price"], errors="coerce").notna()].copy()
    if sub.empty:
        return {"n_rows": float(len(df)), "n_with_actual": 0.0}
    actual = sub["actual_sale_price"].astype(float)
    pred = sub["pred_price"].astype(float)
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    medae = float(np.median(np.abs(err)))
    mape = float(np.mean(np.abs(err / actual)) * 100)
    out: dict[str, float] = {
        "n_rows": float(len(df)),
        "n_with_actual": float(len(sub)),
        "mae_dollar": mae,
        "medae_dollar": medae,
        "mape_pct": mape,
        "rmse_dollar": float(np.sqrt(np.mean(err**2))),
    }
    if (
        "pred_price_low" in sub.columns
        and "pred_price_high" in sub.columns
        and sub["pred_price_low"].notna().any()
    ):
        lo = pd.to_numeric(sub["pred_price_low"], errors="coerce")
        hi = pd.to_numeric(sub["pred_price_high"], errors="coerce")
        ok = lo.notna() & hi.notna()
        if ok.any():
            cov = ((actual[ok] >= lo[ok]) & (actual[ok] <= hi[ok])).mean()
            out["interval_coverage_pct"] = float(cov * 100)
            out["n_rows_for_interval_coverage"] = float(ok.sum())
    if "actual_inside_interval" in sub.columns:
        ins = sub["actual_inside_interval"].astype(str).str.lower().isin(("true", "1"))
        if ins.any():
            out["interval_coverage_from_flag_pct"] = float(ins.mean() * 100)
    return out
