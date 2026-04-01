#!/usr/bin/env python3
"""Load a saved pipeline and write scored CSV (pred_log_price, pred_price in dollars)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import pandas as pd

from housingprices.train import score_dataframe


def main() -> None:
    p = argparse.ArgumentParser(description="Score listings with a trained .joblib pipeline.")
    p.add_argument("--model", type=Path, required=True, help="Path to *_ridge_logprice.joblib")
    p.add_argument("--input", type=Path, required=True, help="Cleaned homes CSV (same schema as training).")
    p.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Training manifest JSON (adds pred_price_low/high using sigma_log_train_residual).",
    )
    args = p.parse_args()

    pipe = joblib.load(Path(args.model).resolve())
    df = pd.read_csv(Path(args.input).resolve())
    sigma = None
    z = 1.96
    if args.manifest:
        m = json.loads(Path(args.manifest).resolve().read_text(encoding="utf-8"))
        s = m.get("sigma_log_train_residual")
        if s is not None and float(s) > 0:
            sigma = float(s)
        z = float(m.get("interval", {}).get("z", 1.96))
    scored = score_dataframe(pipe, df, sigma_log=sigma, z=z)
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    print(f"Wrote {len(scored)} rows to {out}")


if __name__ == "__main__":
    main()
