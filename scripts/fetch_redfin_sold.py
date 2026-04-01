#!/usr/bin/env python3
"""
Download sold listings (last 6 months, house+condo+townhouse) for a ZIP from redfin.com.

Improvements over archive/notebooks/redfin_sold_homes.ipynb:
  - CLI (zip, output path, delays, max pages, resume)
  - Retries with backoff on 429/5xx
  - Timeouts on every request
  - Structured logging
  - Safer HTML parsing (missing attributes, bounds checks)
  - Optional resume: skip URLs already present in an existing CSV
  - Collects parse/fetch errors without stopping the whole run

Legal: automated access may violate Redfin's terms; use licensed feeds for a commercial product.
Install: pip install -r requirements-scrape.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from housingprices.redfin_scrape import (
    build_session,
    fetch_all_listing_paths,
    fetch_listing_records,
)

DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def _urls_in_csv(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    df = pd.read_csv(path, dtype=str)
    if "link" not in df.columns:
        return set()
    return set(df["link"].dropna().astype(str))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--zip", required=True, help="ZIP code, e.g. 97062")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: data/{zip}_homes_data.csv)",
    )
    p.add_argument("--delay", type=float, default=1.25, help="Seconds between HTTP requests")
    p.add_argument("--max-pages", type=int, default=None, help="Cap search pagination (debug)")
    p.add_argument("--limit", type=int, default=None, help="Max listings to fetch after discovery")
    p.add_argument(
        "--resume",
        action="store_true",
        help="If --out exists, skip rows whose link is already in that file",
    )
    p.add_argument("--user-agent", default=DEFAULT_UA, help="Request User-Agent header")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    zipcode = args.zip.strip()
    out = args.out or (ROOT / "data" / f"{zipcode}_homes_data.csv")
    out = Path(out).resolve()

    session = build_session(user_agent=args.user_agent, timeout=args.timeout)
    paths = fetch_all_listing_paths(session, zipcode, delay_s=args.delay, max_pages=args.max_pages)
    if args.limit is not None:
        paths = paths[: args.limit]
    logging.info("Discovered %s listing URLs", len(paths))

    skip: set[str] = set()
    if args.resume:
        skip = _urls_in_csv(out)
        if skip:
            logging.info("Resume: skipping %s URLs already in %s", len(skip), out)

    rows, errors = fetch_listing_records(session, paths, delay_s=args.delay, skip_urls=skip)

    new_df = pd.DataFrame(rows)
    if args.resume and out.is_file() and not new_df.empty:
        old = pd.read_csv(out)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["link"], keep="last", inplace=True)
        df = combined
    elif args.resume and out.is_file() and new_df.empty:
        df = pd.read_csv(out)
    else:
        df = new_df

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logging.info("Wrote %s rows to %s", len(df), out)

    if errors:
        err_path = out.with_suffix(".errors.log")
        err_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        logging.warning("%s errors logged to %s", len(errors), err_path)


if __name__ == "__main__":
    main()
