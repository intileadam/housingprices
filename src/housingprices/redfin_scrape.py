"""
HTML helpers for sold-listing pages on redfin.com.

Redfin markup changes often; this mirrors the notebook's selectors. Scraping may violate
site terms — prefer licensed data for production.
"""

from __future__ import annotations

import json
import logging
import re
import time
from calendar import month_abbr
from typing import Any, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

LIST_URL_TEMPLATE = (
    "https://www.redfin.com/zipcode/{zipcode}/filter/"
    "sort=hi-sale-date,property-type=house+condo+townhouse,include=sold-6mo/page-{page}"
)

def build_session(
    *,
    user_agent: str,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 0.5,
) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml",
        }
    )
    r = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status_forcelist=(429, 500, 502, 503, 504),
        backoff_factor=backoff,
        allowed_methods=("GET",),
    )
    a = HTTPAdapter(max_retries=r)
    s.mount("https://", a)
    s.mount("http://", a)
    s.request_timeout = timeout  # type: ignore[attr-defined]
    return s


def session_get(session: requests.Session, url: str) -> requests.Response:
    timeout = getattr(session, "request_timeout", 30.0)
    return session.get(url, timeout=timeout)


def collect_listing_paths_from_search_page(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/"):
            continue
        # Notebook heuristic: path ending with 8-digit id (fragile but preserved)
        if len(href) >= 8 and href.rstrip("/")[-8:].isdigit() and "/home/" in href:
            if href not in seen:
                seen.add(href)
                out.append(href)
    return out


def fetch_all_listing_paths(
    session: requests.Session,
    zipcode: str,
    *,
    delay_s: float,
    max_pages: Optional[int] = None,
) -> list[str]:
    page = 1
    all_paths: list[str] = []
    seen: set[str] = set()
    while True:
        if max_pages is not None and page > max_pages:
            break
        url = LIST_URL_TEMPLATE.format(zipcode=zipcode, page=page)
        logger.info("GET search page %s", url)
        r = session_get(session, url)
        if r.status_code != 200:
            logger.warning("Search page %s status %s", page, r.status_code)
            break
        paths = collect_listing_paths_from_search_page(r.text)
        new = [p for p in paths if p not in seen]
        if not new:
            logger.info("No new listing links on page %s; stopping pagination.", page)
            break
        for p in new:
            seen.add(p)
            all_paths.append(p)
        page += 1
        if delay_s > 0:
            time.sleep(delay_s)
    return all_paths


def _extract_ld_json_geo(soup: BeautifulSoup) -> tuple[Optional[float], Optional[float]]:
    """Best-effort lat/lon from JSON-LD ``SingleFamilyResidence`` / ``Apartment`` blocks."""
    lat, lon = None, None
    for script in soup.find_all("script", type=lambda t: t and "ld+json" in t):
        raw = script.string or script.get_text()
        if not raw or not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            geo = item.get("geo")
            if isinstance(geo, dict):
                try:
                    lat = float(geo.get("latitude"))
                    lon = float(geo.get("longitude"))
                    return lat, lon
                except (TypeError, ValueError):
                    continue
    return lat, lon


def _parse_us_date(text: str) -> Optional[str]:
    """Normalize to ISO date string YYYY-MM-DD if a US-style date is found."""
    if not text:
        return None
    t = text.strip()
    m = re.search(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),\s*(\d{4})\b",
        t,
        re.I,
    )
    if m:
        mon = m.group(1).title()[:3]
        try:
            mi = list(month_abbr).index(mon)
        except ValueError:
            mi = 0
        if mi:
            return f"{m.group(3)}-{mi:02d}-{int(m.group(2)):02d}"
    m2 = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m2:
        return m2.group(0)
    return None


def _safe_span_text(span) -> str:
    inner = span.find("span")
    if inner is not None and inner.text:
        return inner.text.strip()
    return span.get_text(strip=True)


def parse_listing_page(soup: BeautifulSoup, url: str) -> Optional[dict[str, Any]]:
    """
    Extract the same fields as archive/notebooks/redfin_sold_homes.ipynb (best effort).
    Returns None if no street address node found.
    """
    record: dict[str, Any] = {"link": url}
    current: Optional[dict[str, Any]] = None

    divs = soup.find_all("div")
    for i, div in enumerate(divs):
        classes = div.get("class") or []
        if "street-address" in classes:
            title = div.get("title") or div.get_text(strip=True)
            if not title:
                continue
            current = record
            current["Address"] = title.strip()

        if current is None:
            continue

        if "beds-section" in classes:
            test_id = div.get("data-rf-test-id") or ""
            if "price" in test_id:
                inner = div.find("div")
                if inner and inner.text:
                    current["price"] = inner.text.strip()
            else:
                if i + 1 < len(divs):
                    current["beds"] = divs[i + 1].get_text(strip=True)
        if "baths-section" in classes:
            if i + 1 < len(divs):
                current["baths"] = divs[i + 1].get_text(strip=True)
        if "sqft-section" in classes:
            sp = div.find("span")
            if sp and sp.text:
                current["sqft"] = sp.text.strip()

        if "super-group-content" in classes:
            for span in div.find_all("span"):
                scls = span.get("class") or []
                if "entryItemContent" not in scls:
                    continue
                text = span.get_text()
                val = _safe_span_text(span)
                if "Parking Features" in text:
                    current["parking_features"] = val
                elif "Parking Total" in text:
                    current["parking_total"] = val
                elif "Garage Type" in text:
                    current["garage_type"] = val
                elif "Garage Spaces" in text:
                    current["garage_spaces"] = val
                elif "Hot Water Description" in text:
                    current["hot_water_desc"] = val
                elif "Fireplace YN" in text:
                    current["fireplace_yn"] = val
                elif "Basement" in text:
                    current["basement"] = val
                elif "Lot Size Square Feet" in text:
                    current["lot_size_sqft"] = val
                elif "Property Type" in text:
                    current["property_type"] = val
                elif "Year Built" in text:
                    current["yr_built"] = val
                elif "Fuel Description" in text:
                    current["fuel_desc"] = val
                elif "Has HOA" in text:
                    current["has_hoa"] = val
                elif "View YN" in text:
                    current["view_yn"] = val
                elif "Fireplaces Total" in text:
                    current["fireplace_total"] = val
                elif "Main Level Area Total" in text:
                    current["main_level_area"] = val
                elif "Sewer" in text:
                    current["sewer"] = val
                elif "Cooling YN" in text:
                    current["cooling_yn"] = val
                elif "Senior Community YN" in text:
                    current["senior_community_yn"] = val
                elif "Lot Size" in text and "Square Feet" not in text:
                    current["lot_size"] = val
                elif "Style" in text and "Cooling" not in text:
                    current["style"] = val
                elif "Year Renovated" in text:
                    current["yr_renovated"] = val
                elif "County" in text:
                    current["county"] = val
                elif "New Construction YN" in text:
                    current["new_construction_yn"] = val
                elif "Stories" in text:
                    current["stories"] = val
                elif "Roof" in text:
                    current["roof"] = val
                elif "Days on Market" in text or "Time on Redfin" in text:
                    dm = re.search(r"(\d+)\s*days?", val, re.I)
                    if dm:
                        current["days_on_market"] = dm.group(1)
                    elif val.isdigit():
                        current["days_on_market"] = val
                elif "Sold" in text and "Bought" not in text:
                    d = _parse_us_date(val) or _parse_us_date(text)
                    if d:
                        current["sale_date"] = d

    lat, lon = _extract_ld_json_geo(soup)
    if lat is not None:
        record["latitude"] = lat
    if lon is not None:
        record["longitude"] = lon

    if "Address" not in record:
        return None
    return record


def fetch_listing_records(
    session: requests.Session,
    paths: list[str],
    *,
    delay_s: float,
    base: str = "https://www.redfin.com",
    skip_urls: Optional[set[str]] = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    skip_urls = skip_urls or set()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        url = urljoin(base, path)
        if url in skip_urls:
            logger.debug("skip (resume) %s", url)
            continue
        logger.info("GET listing %s", url)
        try:
            r = session_get(session, url)
        except requests.RequestException as e:
            errors.append(f"{url} request error: {e}")
            if delay_s > 0:
                time.sleep(delay_s)
            continue
        if r.status_code != 200:
            errors.append(f"{url} status {r.status_code}")
            if delay_s > 0:
                time.sleep(delay_s)
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        rec = parse_listing_page(soup, url)
        if rec is None:
            errors.append(f"{url} parse: no street-address")
        else:
            rows.append(rec)
        if delay_s > 0:
            time.sleep(delay_s)
    return rows, errors
