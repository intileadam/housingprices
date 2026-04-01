"""Run manifests: data hashes, config, library versions."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import sklearn

    _SKLEARN = sklearn.__version__
except ImportError:
    _SKLEARN = "unknown"

try:
    import numpy as np

    _NUMPY = np.__version__
except ImportError:
    _NUMPY = "unknown"

try:
    import pandas as pd

    _PANDAS = pd.__version__
except ImportError:
    _PANDAS = "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    *,
    data_path: Path,
    zipcode: str,
    winsor_quantile: float,
    winsor_threshold_price: float,
    n_train_rows_raw: int,
    n_train_rows_after_winsor: int,
    n_test_rows: int,
    random_state: int,
    ridge_alphas: list[float],
    chosen_alpha: Optional[float],
    metrics: Mapping[str, Any],
    model_version: str,
    notes: Optional[str] = None,
    temporal_validation: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    data_path = Path(data_path).resolve()
    m: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "python": sys.version.split()[0],
        "numpy": _NUMPY,
        "pandas": _PANDAS,
        "sklearn": _SKLEARN,
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path) if data_path.is_file() else None,
        "zipcode": zipcode,
        "winsor": {
            "quantile": winsor_quantile,
            "threshold_price_fit_on_train_only": winsor_threshold_price,
            "description": "Training rows use only listings with price < quantile(price on raw train split). "
            "Threshold is computed from the train split before fitting the preprocessor (no test leakage).",
        },
        "splits": {
            "random_state": random_state,
            "n_train_raw": n_train_rows_raw,
            "n_train_after_winsor": n_train_rows_after_winsor,
            "n_test": n_test_rows,
        },
        "ridge": {"alphas_searched": list(ridge_alphas), "chosen_alpha": chosen_alpha},
        "metrics": dict(metrics),
        "temporal_validation": temporal_validation
        if temporal_validation is not None
        else {
            "status": "not_available",
            "reason": "No usable sale_date column; add sale_date to enable chronological train/test.",
        },
    }
    if notes:
        m["notes"] = notes
    m["data_licensing"] = (
        "Scraped or exported third-party listing data may be restricted by terms of use. "
        "For a commercial product, plan on licensed MLS or vendor feeds."
    )
    return m


def write_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
