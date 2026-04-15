import os
import logging
from typing import Optional, List, Dict, Any

import requests

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"


def fetch_agri_news(
    state: Optional[str],
    crop: Optional[str],
    page_size: int = 10,
    timeout: float = 8.0,
) -> List[Dict[str, Any]]:
    """Fetch recent agri news filtered by state + crop from NewsAPI.

    Returns a trimmed list of articles (title, description, url, urlToImage,
    publishedAt, source). Returns [] on any failure so callers can treat it
    as non-blocking.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        logger.warning("[news] NEWSAPI_KEY not set; skipping news fetch")
        return []

    if not state and not crop:
        logger.info("[news] no state/crop provided; skipping news fetch")
        return []

    query = "agriculture " + " ".join(part for part in [state, crop] if part).strip()
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    try:
        prep = requests.Request("GET", NEWSAPI_URL, params={**params, "apiKey": "***"}).prepare()
        logger.info(f"[news] GET {prep.url}")
        resp = requests.get(NEWSAPI_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"[news] fetch failed: {e}")
        return []

    if data.get("status") != "ok":
        logger.warning(f"[news] API returned non-ok status: {data.get('status')} / {data.get('message')}")
        return []

    articles = data.get("articles", []) or []
    trimmed = []
    for a in articles:
        trimmed.append({
            "title": a.get("title"),
            "description": a.get("description"),
            "url": a.get("url"),
            "urlToImage": a.get("urlToImage"),
            "publishedAt": a.get("publishedAt"),
            "source": (a.get("source") or {}).get("name"),
        })
    logger.info(f"[news] returned {len(trimmed)} articles")
    return trimmed
