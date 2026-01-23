# services/tickers.py

import time
import json
from typing import Optional

from google.cloud import storage


# ---------------------------------------------------------
# IN-MEMORY CACHE
# ---------------------------------------------------------

_TICKER_CACHE = {
    "data": None,
    "timestamp": 0,
}

CACHE_TTL = 60 * 60 * 24  # 24 hours


# ---------------------------------------------------------
# GCS BUCKET LOADER
# ---------------------------------------------------------

def _get_bucket():
    """
    Lazy-loads the GCS bucket client.
    """
    client = storage.Client()
    return client.bucket("detective-app-67ffd.appspot.com")


def _load_tickers_from_gcs() -> Optional[list]:
    """
    Loads tickers.json from Cloud Storage.
    Returns a Python list or None if unavailable.
    """
    try:
        bucket = _get_bucket()
        blob = bucket.blob("tickers/tickers.json")
        content = blob.download_as_text()
        return json.loads(content)
    except Exception as e:
        print("Error loading tickers from GCS:", e)
        return None


# ---------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------

def get_tickers_cached():
    """
    Returns cached tickers if fresh.
    Otherwise loads from GCS and updates cache.
    """

    global _TICKER_CACHE

    # Return cached version if still valid
    if (
        _TICKER_CACHE["data"] is not None
        and (time.time() - _TICKER_CACHE["timestamp"] < CACHE_TTL)
    ):
        return _TICKER_CACHE["data"]

    # Load from GCS
    tickers = _load_tickers_from_gcs()
    if tickers is None:
        # If GCS fails but we have old cache, return it
        if _TICKER_CACHE["data"] is not None:
            return _TICKER_CACHE["data"]
        # Otherwise return empty list
        return []

    # Update cache
    _TICKER_CACHE["data"] = tickers
    _TICKER_CACHE["timestamp"] = time.time()

    return tickers