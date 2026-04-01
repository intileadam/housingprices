#!/usr/bin/env python3
"""Print aggregate error metrics for ledger rows that have actual_sale_price filled in."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from housingprices.ledger import load_ledger, reconcile


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize prediction vs actual from the ledger CSV.")
    p.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "predictions" / "ledger.csv",
        help="Path to ledger.csv",
    )
    p.add_argument("--run-id", help="If set, filter to this run_id only.")
    args = p.parse_args()

    path = Path(args.ledger).resolve()
    if args.run_id:
        df = load_ledger(path)
        if df.empty:
            print(json.dumps({"error": "empty ledger"}, indent=2))
            return
        df = df[df["run_id"].astype(str) == str(args.run_id)]
        tmp = path.parent / f".reconcile_tmp_{args.run_id}.csv"
        df.to_csv(tmp, index=False)
        stats = reconcile(tmp)
        tmp.unlink(missing_ok=True)
    else:
        stats = reconcile(path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
