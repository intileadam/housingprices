#!/usr/bin/env python3
"""Train pipeline (Ridge by default; optional challengers), write model + manifest + ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from housingprices import __version__
from housingprices.ledger import append_prediction_rows, build_rows_from_scored_df, make_run_id
from housingprices.provenance import build_manifest, sha256_file
from housingprices.train import RANDOM_STATE, save_artifacts, score_dataframe, train_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Train log-price model on cleaned home CSV.")
    p.add_argument("--zip", required=True, help="ZIP code (used in default paths and manifest).")
    p.add_argument(
        "--data",
        type=Path,
        help="Path to cleaned CSV (default: data/{zip}_homes_data_cleaned.csv).",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--winsor-quantile", type=float, default=0.95)
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument(
        "--random-split",
        "--force-random-split",
        action="store_true",
        dest="force_random_split",
        help="Use random holdout even when sale_date supports chronological split.",
    )
    p.add_argument(
        "--min-time-parse-fraction",
        type=float,
        default=0.45,
        help="Minimum fraction of rows with parseable sale_date to use chronological split.",
    )
    p.add_argument(
        "--compare-challengers",
        action="store_true",
        help="Also fit ElasticNet and HistGradientBoosting; log metrics in manifest.",
    )
    p.add_argument(
        "--promote-challenger",
        action="store_true",
        help="If chronological split and a challenger beats Ridge on test MAE ($), save that model.",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "models", help="Directory for .joblib + manifest.")
    p.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Write scored rows (pred_price in dollars) to this path.",
    )
    p.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "predictions" / "ledger.csv",
        help="Append prediction run to this ledger CSV.",
    )
    p.add_argument("--no-ledger", action="store_true")
    args = p.parse_args()

    zipcode = args.zip.strip()
    data_path = args.data or (ROOT / "data" / f"{zipcode}_homes_data_cleaned.csv")
    data_path = Path(data_path).resolve()
    if not data_path.is_file():
        raise SystemExit(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    pipe, meta = train_pipeline(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        winsor_quantile=args.winsor_quantile,
        force_random_split=args.force_random_split,
        compare_challengers=args.compare_challengers,
        promote_challenger=args.promote_challenger,
        min_time_split_parse_fraction=args.min_time_parse_fraction,
    )

    out_dir = Path(args.out_dir).resolve()
    model_path = out_dir / f"{zipcode}_ridge_logprice.joblib"
    manifest_path = out_dir / f"{zipcode}_manifest.json"

    data_sha = sha256_file(data_path)
    split_meta = meta.get("split", {})
    if split_meta.get("split") == "chronological_by_sale_date":
        temporal_validation = {
            "status": "time_based_default",
            "sale_date_column": "sale_date",
            "policy": "Chronological split is used by default when sale_date parses for enough rows.",
            **{k: v for k, v in split_meta.items() if k != "split"},
        }
    else:
        temporal_validation = {
            "status": "random_holdout",
            "random_state": split_meta.get("random_state", args.random_state),
            "chronological_not_used_reason": split_meta.get("chronological_not_used_reason"),
            "note": "Add sale_date (e.g. from fetch_redfin_sold) to enable default time-based validation.",
        }
    full_manifest = build_manifest(
        data_path=data_path,
        zipcode=zipcode,
        winsor_quantile=args.winsor_quantile,
        winsor_threshold_price=meta["winsor_threshold_price"],
        n_train_rows_raw=meta["n_train_raw"],
        n_train_rows_after_winsor=meta["n_train_after_winsor"],
        n_test_rows=meta["n_test"],
        random_state=args.random_state,
        ridge_alphas=meta["alphas_searched"],
        chosen_alpha=meta.get("chosen_alpha"),
        metrics=meta["metrics"],
        model_version=__version__,
        temporal_validation=temporal_validation,
    )
    full_manifest["primary_model"] = meta["primary_model"]
    full_manifest["sigma_log_train_residual"] = meta["sigma_log_train_residual"]
    full_manifest["interval"] = {
        "method": meta["interval_method"],
        "z": meta["interval_z"],
        "note": meta["interval_note"],
    }
    if meta.get("challenger_comparison"):
        full_manifest["challenger_comparison"] = meta["challenger_comparison"]

    save_artifacts(pipe, full_manifest, model_path=model_path, manifest_path=manifest_path)
    print(f"Wrote model: {model_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Primary model: {meta['primary_model']}")
    if meta.get("chosen_alpha") is not None:
        print(f"Ridge chosen alpha: {meta['chosen_alpha']}")
    print("Test metrics:", json.dumps(meta["metrics"], indent=2))

    sigma = meta.get("sigma_log_train_residual")
    scored = score_dataframe(pipe, df, sigma_log=sigma if sigma and sigma > 0 else None)

    if args.predictions_csv:
        pred_path = Path(args.predictions_csv).resolve()
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(pred_path, index=False)
        print(f"Wrote predictions: {pred_path}")

    if not args.no_ledger:
        run_id = make_run_id()
        rows = build_rows_from_scored_df(
            scored,
            run_id=run_id,
            model_version=__version__,
            data_file_sha256=data_sha,
            zipcode=zipcode,
            winsor_threshold=meta["winsor_threshold_price"],
            interval_method=meta.get("interval_method", ""),
            interval_z=str(meta.get("interval_z", "")),
            interval_sigma_log=str(sigma) if sigma is not None else "",
        )
        ledger_path = Path(args.ledger).resolve()
        append_prediction_rows(ledger_path, rows)
        print(f"Appended {len(rows)} rows to ledger: {ledger_path} (run_id={run_id})")


if __name__ == "__main__":
    main()
