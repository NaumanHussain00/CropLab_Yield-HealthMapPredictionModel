import os
import time
import logging
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any

import requests

logger = logging.getLogger(__name__)

GOVDATA_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
AGMARKNET_URL = "https://api.agmarknet.gov.in/v1/price-trend/wholesale-prices-monthly"

# agmarknet numeric IDs (official).
STATE_ID_MAP = {
    "andaman and nicobar islands": 1,
    "andhra pradesh": 2,
    "arunachal pradesh": 3,
    "assam": 4,
    "bihar": 5,
    "chandigarh": 6,
    "chhattisgarh": 7,
    "dadra and nagar haveli and daman and diu": 8,
    "delhi": 9,
    "goa": 10,
    "gujarat": 11,
    "haryana": 12,
    "himachal pradesh": 13,
    "jammu and kashmir": 14,
    "jharkhand": 15,
    "karnataka": 16,
    "kerala": 17,
    "ladakh": 18,
    "lakshadweep": 19,
    "madhya pradesh": 20,
    "maharashtra": 21,
    "manipur": 22,
    "meghalaya": 23,
    "mizoram": 24,
    "nagaland": 25,
    "odisha": 26,
    "puducherry": 27,
    "punjab": 28,
    "rajasthan": 29,
    "sikkim": 30,
    "tamil nadu": 31,
    "telangana": 32,
    "tripura": 33,
    "uttar pradesh": 34,
    "uttarakhand": 35,
    "west bengal": 36,
}

COMMODITY_ID_MAP = {
    "wheat": 1,
    "rice": 3,
    "baby corn": 459,
    "soyabeans": 13,
    "soybean": 13,
    "cotton": 15,
    "sugarcane": 122,
    "potato": 24,
    "tomato": 65,
    "onion": 23,
    "cabbage": 126,
    "carrot": 125,
    "beans": 80,
    "green peas": 46,
    "sunflower": 14,
    "barley": 29,
}

COMMODITY_ID_DEFAULT = 1  # wheat


def _fetch_govdata_for_date(state: str, crop: Optional[str], arrival_date: date, timeout: float) -> Dict[str, Any]:
    api_key = os.getenv("GOVDATA_API_KEY")
    if not api_key:
        return {"date": arrival_date.isoformat(), "records": [], "error": "GOVDATA_API_KEY not set"}

    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 1000,
        "filters[state.keyword]": state.title(),
        "filters[arrival_date]": arrival_date.strftime("%d/%m/%Y"),
    }
    if crop:
        params["filters[commodity]"] = crop.title()

    try:
        prep = requests.Request("GET", GOVDATA_URL, params={**params, "api-key": "***"}).prepare()
        logger.info(f"[rates] govdata GET {prep.url}")
        resp = requests.get(GOVDATA_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return {
            "date": arrival_date.isoformat(),
            "records": data.get("records", []) or [],
        }
    except Exception as e:
        logger.warning(f"[rates] govdata fetch failed for {arrival_date}: {e}")
        return {"date": arrival_date.isoformat(), "records": [], "error": str(e)}


def fetch_govdata_last_n_days(
    state: str,
    crop: Optional[str],
    days: int = 7,
    timeout: float = 15.0,
    stagger_ms: int = 500,
) -> List[Dict[str, Any]]:
    """Fetch govdata mandi prices for the last `days` calendar days.

    Requests are dispatched on a background pool but launches are staggered by
    `stagger_ms` milliseconds so we don't hammer data.gov.in (which rate-limits
    bursts). With days=7 and stagger_ms=500 the full fan-out takes ~3s to
    *start* all requests; they still complete in parallel after that.
    """
    if not state:
        return []
    today = date.today()
    dates = [today - timedelta(days=i) for i in range(days)]

    gap = max(stagger_ms, 0) / 1000.0
    with ThreadPoolExecutor(max_workers=min(days, 8)) as pool:
        futures = []
        for d in dates:
            futures.append(pool.submit(_fetch_govdata_for_date, state, crop, d, timeout))
            if gap:
                time.sleep(gap)
        results = [f.result() for f in futures]
    return results


# Backwards-compat alias
def fetch_govdata_last_3_days(state: str, crop: Optional[str], timeout: float = 15.0) -> List[Dict[str, Any]]:
    return fetch_govdata_last_n_days(state, crop, days=7, timeout=timeout)


def fetch_agmarknet(state: Optional[str], crop: Optional[str], timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    """Fetch agmarknet monthly wholesale price trend (district-wise) for current month."""
    if not state or not crop:
        logger.info("[rates] agmarknet requires state + crop; skipping")
        return None

    state_id = STATE_ID_MAP.get(state.strip().lower())
    if state_id is None:
        logger.warning(f"[rates] agmarknet: unmapped state='{state}'")
        return None

    commodity_id = COMMODITY_ID_MAP.get(crop.strip().lower(), COMMODITY_ID_DEFAULT)
    if crop.strip().lower() not in COMMODITY_ID_MAP:
        logger.info(f"[rates] agmarknet: unmapped crop='{crop}', falling back to default commodity_id={COMMODITY_ID_DEFAULT}")

    today = date.today()
    params = {
        "report_mode": "Districtwise",
        "commodity": commodity_id,
        "year": today.year,
        "month": today.month,
        "state": state_id,
        "district": 0,
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.agmarknet.gov.in/",
        "Origin": "https://www.agmarknet.gov.in",
    }
    try:
        prep = requests.Request("GET", AGMARKNET_URL, params=params).prepare()
        logger.info(f"[rates] agmarknet GET {prep.url}")
        resp = requests.get(AGMARKNET_URL, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"[rates] agmarknet fetch failed: {e}")
        return None
