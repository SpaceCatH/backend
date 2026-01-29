# services/alpha_vantage.py

import os
import requests
from fastapi import HTTPException

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    raise RuntimeError("ALPHA_VANTAGE_API_KEY environment variable is not set.")

API_URL = "https://www.alphavantage.co/query"

def fetch_eod_data(ticker: str, limit: int = 120):
    """
    Fetches daily OHLC data from Alpha Vantage (free tier).
    Returns a list of candles sorted by date ascending.
    """

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "compact",
    }

    try:
        resp = requests.get(API_URL, params=params, timeout=10)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Network error: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Alpha Vantage request failed")

    data = resp.json()
    print("ALPHA RAW:", data)

    # --- Pass through Alpha Vantage messages verbatim ---

# Rate limit or general API notice
if "Note" in data:
    raise HTTPException(status_code=429, detail=data["Note"])

# Premium endpoint or temporary outage message
if "Information" in data:
    raise HTTPException(status_code=429, detail=data["Information"])

# Invalid symbol or malformed request
if "Error Message" in data:
    raise HTTPException(status_code=400, detail=data["Error Message"])

# Missing or empty time series
if "Time Series (Daily)" not in data or not data["Time Series (Daily)"]:
    raise HTTPException(status_code=400, detail=f"Alpha Vantage returned no daily data for {ticker}")

ts = data["Time Series (Daily)"]

    candles = []
    for date_str, values in ts.items():
        try:
            candles.append(
                {
                    "date": date_str,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                }
            )
        except Exception:
            continue

    if not candles:
        raise HTTPException(status_code=400, detail=f"No valid candles for {ticker}")

    candles.sort(key=lambda x: x["date"])
    return candles[-limit:]
