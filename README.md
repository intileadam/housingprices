# housingprices

Local hedonic models for home sale prices: a scikit-learn **Pipeline** that predicts **log(price)** with **RidgeCV** by default (optional **ElasticNetCV** and **HistGradientBoostingRegressor** for comparison or promotion), preprocessing aligned with the archived v7 notebook, and **time-aware validation** when `sale_date` is usable.

## Repository layout

| Path | Purpose |
|------|--------|
| `src/housingprices/` | `HousingPreprocessor`, `train.py` (fit + `score_dataframe`), `challengers.py`, `redfin_scrape.py`, `ledger.py`, `provenance.py` |
| `scripts/` | `train_model.py`, `score_batch.py`, `fetch_redfin_sold.py`, `reconcile_ledger.py` |
| `home_data_cleaning.ipynb` | Active notebook: raw → cleaned CSV matching the training schema |
| `data/` | Raw and cleaned CSVs (e.g. `{zip}_homes_data.csv`, `{zip}_homes_data_cleaned.csv`) — often gitignored |
| `models/` | `{zip}_ridge_logprice.joblib` plus `{zip}_manifest.json` |
| `predictions/` | Scored CSVs from training/scoring and append-only `ledger.csv` |

### Archive (legacy)

Older experiments and one-off exports live under **`archive/`** so the root stays focused on the current pipeline:

- **`archive/notebooks/`** — `v1`–`v7` MLR / Ridge notebooks and `redfin_sold_homes.ipynb` (superseded by `scripts/fetch_redfin_sold.py` and `redfin_scrape.py`).
- **`archive/predictions/`** — Notebook-era `to_csv` prediction dumps; current scoring uses `scripts/score_batch.py` (and optional `--predictions-csv` on train).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Core stack: **numpy**, **pandas**, **scikit-learn**, **joblib**, **scipy** (pinned in `requirements.txt`).

Optional Redfin fetcher:

```bash
pip install -r requirements-scrape.txt
```

Scripts prepend `src/` to `sys.path`; run from the repo root or pass absolute paths.

## Typical workflow

1. **Data**  
   Use your own export or `scripts/fetch_redfin_sold.py`, then clean in `home_data_cleaning.ipynb` to produce `data/{zip}_homes_data_cleaned.csv` with columns expected by `preprocess` (e.g. `beds`, bath-related fields, `sqft`, categoricals in `CAT_COLS`, `price`, and ideally **`sale_date`** for chronological holdout).

2. **Train** — saves the fitted pipeline, a JSON manifest, optionally a scored CSV, and appends the ledger:

   ```bash
   python scripts/train_model.py --zip 97062
   ```

   Notable flags: `--data`, `--compare-challengers`, `--promote-challenger` (only with chronological split: pick best test MAE in $ among Ridge / ElasticNet / HGBR), `--force-random-split`, `--min-time-parse-fraction`, `--predictions-csv`, `--no-ledger`, `--ledger`, `--out-dir`.

   The on-disk model file is always named `{zip}_ridge_logprice.joblib`; check manifest **`primary_model`** (`ridge`, `elasticnet`, or `hist_gradient_boosting`) to see what was actually saved when promotion is on.

3. **Score** — same schema as training; optional **prediction intervals** if you pass the training manifest:

   ```bash
   python scripts/score_batch.py \
     --model models/97062_ridge_logprice.joblib \
     --manifest models/97062_manifest.json \
     --input data/97062_homes_data_cleaned.csv \
     --output predictions/97062_scored.csv
   ```

   `--manifest` supplies `sigma_log_train_residual` and interval `z` so outputs include `pred_log_low` / `pred_log_high` and `pred_price_low` / `pred_price_high` (log-space symmetric normal approximation, then `exp`). Without `--manifest`, you get point predictions only.

4. **Ledger** — after backfilling realized prices, summarize errors (and interval coverage when bounds exist):

   ```bash
   python scripts/reconcile_ledger.py
   python scripts/reconcile_ledger.py --run-id <uuid>
   ```

## Manifest (`*_manifest.json`)

Each training run writes a manifest beside the `.joblib` file, including:

- **Provenance**: `created_at_utc`, `model_version`, Python / numpy / pandas / sklearn versions, `data_path`, `data_sha256`
- **Winsorization**: quantile and train-only price threshold (no test leakage)
- **Splits**: row counts; **`temporal_validation`** documents chronological vs random holdout
- **Ridge**: alphas searched and chosen alpha (when Ridge is primary)
- **Metrics**: test R²/MAE/RMSE in log and dollars, simple baselines, high-price counts vs winsor threshold
- **`primary_model`**, **`sigma_log_train_residual`**, **`interval`**: method, `z`, and note for scoring intervals
- **`data_licensing`**: short reminder that scraped or exported listing data may be terms-restricted
- With **`--compare-challengers`**: optional **`challenger_comparison`** block on test MAE ($)

## Redfin scraping (optional)

```bash
python scripts/fetch_redfin_sold.py --zip 97062 --resume
```

Default output: `data/{zip}_homes_data.csv`. Automated access may conflict with site terms; use licensed feeds for production (see manifest `data_licensing`).

## Model behavior (short)

- **Target**: log(price); exports include `pred_price` = exp(pred log).
- **Default split**: chronological by `sale_date` when enough dates parse; otherwise random holdout (override with `--force-random-split`).
- **Winsorization**: training rows capped by train-split price quantile before fitting.
- **Artifact**: one `joblib` pipeline (preprocessor + regressor); load with `joblib.load`.

---

Version: `housingprices.__version__` in `src/housingprices/__init__.py`.
