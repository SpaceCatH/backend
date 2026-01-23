from firebase_client import db
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import time
import json
from typing import List, Optional
from math import floor
from dotenv import load_dotenv

load_dotenv()

_ticker_cache = {
    "data": None,
    "timestamp": 0
}

CACHE_TTL = 60 * 60 * 24  # 24 hours

# -----------------------------
# FastAPI app setup
# -----------------------------

app = FastAPI(
    title="8 EMA Swing Trading Strategy API",
    description="Generates entry/exit points for multiple EMA breakout strategies and scans the S&P 100.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    raise RuntimeError("ALPHA_VANTAGE_API_KEY environment variable is not set.")

# -----------------------------
# Lazy GCS helper
# -----------------------------

def get_bucket():
    from google.cloud import storage
    client = storage.Client()
    return client.bucket("detective-app-67ffd.appspot.com")

# -----------------------------
# S&P 100 universe (static)
# -----------------------------

SP100_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMGN", "AMT", "AMZN", "AVGO",
    "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT",
    "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD", "GE",
    "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ",
    "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT",
    "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE",
    "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX", "SBUX",
    "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP",
    "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM"
]

# -----------------------------
# Firestore ticker normalization
# -----------------------------

def normalize_ticker(ticker: str) -> str:
    return ticker.replace(".", "-")

# -----------------------------
# Pydantic models
# -----------------------------

class StrategyResult(BaseModel):
    strategy: str
    entry: float
    stop_loss: float
    take_profit: float
    shares: int
    risk_per_share: float
    total_risk: float
    total_profit: float
    notes: str
    score: float = 0.0
    is_recommended: bool = False

class StrategyResponse(BaseModel):
    strategies: List[StrategyResult]

class ScanCandidate(BaseModel):
    ticker: str
    best_strategy: str
    best_score: float
    has_simple: bool
    has_swing: bool
    has_retest: bool

class ScanResponse(BaseModel):
    candidates: List[ScanCandidate]

# -----------------------------
# Helper functions
# -----------------------------

def fetch_eod_data(ticker: str, limit: int = 120):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "compact",
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Error fetching data from Alpha Vantage")

    data = resp.json()
    if "Time Series (Daily)" not in data:
        raise HTTPException(status_code=400, detail=f"Could not get EOD data for {ticker}")

    ts = data["Time Series (Daily)"]
    candles = []
    for date_str, values in ts.items():
        candles.append(
            {
                "date": date_str,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
            }
        )

    candles.sort(key=lambda x: x["date"])
    return candles[-limit:]

def calculate_ema(prices: List[float], period: int = 8) -> List[float]:
    if len(prices) < period:
        raise HTTPException(status_code=400, detail="Not enough data to calculate EMA")

    ema_values = []
    k = 2 / (period + 1)

    sma = sum(prices[:period]) / period
    ema_values.extend([None] * (period - 1))
    ema_values.append(sma)

    for price in prices[period:]:
        prev_ema = ema_values[-1]
        current_ema = (price - prev_ema) * k + prev_ema
        ema_values.append(current_ema)

    return ema_values

def compute_atr(candles: List[dict], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None

    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]

    trs = []
    for i in range(1, len(candles)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    if len(trs) < period:
        return None

    recent_trs = trs[-period:]
    atr = sum(recent_trs) / period
    return atr

# -----------------------------
# Scoring helpers
# -----------------------------

def compute_trend_strength(ema_values: List[float]) -> float:
    valid = [e for e in ema_values if e is not None]
    if len(valid) < 20:
        return 0.0

    start = valid[-20]
    end = valid[-1]
    if start <= 0:
        return 0.0

    pct = (end - start) / start

    if pct > 0.05:
        return 3.0
    elif pct > 0.02:
        return 2.0
    elif pct > 0.005:
        return 1.0
    return 0.0

def compute_volatility_compression(candles: List[dict]) -> float:
    if len(candles) < 25:
        return 0.0

    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]

    trs = []
    for i in range(1, len(candles)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    if len(trs) < 20:
        return 0.0

    recent = trs[-20:]
    avg_atr = sum(recent[:-1]) / (len(recent) - 1)
    current_atr = recent[-1]

    if avg_atr == 0:
        return 0.0

    ratio = current_atr / avg_atr

    if ratio < 0.8:
        return 2.0
    elif ratio < 1.0:
        return 1.0
    return 0.0

def compute_pullback_quality(candles: List[dict], ema_values: List[float]) -> float:
    closes = [c["close"] for c in candles]
    if len(closes) < 3 or len(ema_values) < 3:
        return 0.0

    i = len(closes) - 1
    if ema_values[i] is None or ema_values[i - 1] is None or ema_values[i - 2] is None:
        return 0.0

    above_two_ago = closes[i - 2] > ema_values[i - 2]
    near_prev = closes[i - 1] <= ema_values[i - 1] * 1.01
    bounce = closes[i] > ema_values[i]

    if above_two_ago and near_prev and bounce:
        return 3.0
    elif above_two_ago and bounce:
        return 2.0
    elif bounce:
        return 1.0
    return 0.0

# -----------------------------
# ATR-based stop/target config
# -----------------------------

MAX_STOP_PCT = 0.10
ATR_MULT_STOP = 1.5

def build_atr_stop_and_target(entry: float, atr: Optional[float], r_mult: float) -> Optional[tuple]:
    if atr is None or atr <= 0 or entry <= 0:
        return None

    raw_risk = ATR_MULT_STOP * atr
    max_risk = entry * MAX_STOP_PCT
    risk_per_share = min(raw_risk, max_risk)

    if risk_per_share <= 0:
        return None

    stop_loss = entry - risk_per_share
    take_profit = entry + r_mult * risk_per_share

    if stop_loss <= 0 or take_profit <= entry:
        return None

    return stop_loss, take_profit, risk_per_share

# -----------------------------
# Strategy implementations
# -----------------------------

# (unchanged — your strategy functions go here exactly as before)

# -----------------------------
# Strategy endpoint
# -----------------------------

@app.get("/strategy", response_model=StrategyResponse)
def get_strategy(
    ticker: str = Query(...),
    dollars: float = Query(..., gt=0),
    type: str = Query("simple"),
):
    ticker_upper = ticker.upper()
    strategy_type = type.lower()

    candles = fetch_eod_data(ticker_upper)
    closes = [c["close"] for c in candles]
    ema_values = calculate_ema(closes, period=8)

    strategies: List[StrategyResult] = []

    if strategy_type in ("simple", "all"):
        res = simple_ema_breakout(candles, ema_values, dollars)
        if res:
            strategies.append(res)

    if strategy_type in ("swing", "all"):
        res = swing_high_breakout(candles, ema_values, dollars)
        if res:
            strategies.append(res)

    if strategy_type in ("retest", "all"):
        res = retest_breakout(candles, ema_values, dollars)
        if res:
            strategies.append(res)

    if not strategies:
        raise HTTPException(
            status_code=404,
            detail="No valid strategy setups were found for this ticker."
        )

    trend = compute_trend_strength(ema_values)
    vol = compute_volatility_compression(candles)
    pull = compute_pullback_quality(candles, ema_values)

    for s in strategies:
        if s.strategy == "simple":
            s.score = trend
        elif s.strategy == "swing":
            s.score = trend + vol
        elif s.strategy == "retest":
            s.score = trend + pull

    best = max(strategies, key=lambda x: x.score)
    for s in strategies:
        s.is_recommended = (s is best)

    return StrategyResponse(strategies=strategies)

# -----------------------------
# Daily scan endpoint
# -----------------------------

# (unchanged — your daily scan code goes here exactly as before)

# -----------------------------
# Scan endpoint
# -----------------------------

# (unchanged — your scan endpoint code goes here exactly as before)

# -----------------------------
# Tickers endpoint (fixed)
# -----------------------------

@app.get("/tickers")
def get_tickers():
    global _ticker_cache

    # Return cached version if fresh
    if _ticker_cache["data"] and (time.time() - _ticker_cache["timestamp"] < CACHE_TTL):
        return _ticker_cache["data"]

    # Lazy-load from Cloud Storage
    bucket = get_bucket()
    blob = bucket.blob("tickers/tickers.json")
    content = blob.download_as_text()

    data = json.loads(content)

    # Update cache
    _ticker_cache["data"] = data
    _ticker_cache["timestamp"] = time.time()

    return data