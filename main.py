from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from typing import List, Optional
from math import floor
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# FastAPI app setup
# -----------------------------

app = FastAPI(
    title="8 EMA Swing Trading Strategy API",
    description="Generates entry/exit points for multiple EMA breakout strategies.",
    version="1.0.0",
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

MAX_STOP_PCT = 0.10      # 10% max stop distance
ATR_MULT_STOP = 1.5      # 1.5 * ATR for stop


def build_atr_stop_and_target(entry: float, atr: float, r_mult: float) -> Optional[tuple]:
    if atr is None or atr <= 0 or entry <= 0:
        return None

    raw_risk = ATR_MULT_STOP * atr
    max_risk = entry * MAX_STOP_PCT
    risk_per_share = min(raw_risk, max_risk)

    if risk_per_share <= 0:
        return None

    stop_loss = entry - risk_per_share
    take_profit = entry + r_mult * risk_per_share

    if stop_loss <= 0:
        return None

    return stop_loss, take_profit, risk_per_share


# -----------------------------
# Strategy implementations
# -----------------------------

def simple_ema_breakout(candles, ema_values, investment_dollars):
    closes = [c["close"] for c in candles]
    LOOKBACK_LIMIT = 7

    last_idx = len(closes) - 1
    start_idx = max(1, last_idx - LOOKBACK_LIMIT)

    found_idx = None
    for i in range(last_idx, start_idx - 1, -1):
        if ema_values[i] is None or ema_values[i - 1] is None:
            continue
        if closes[i] > ema_values[i] and closes[i - 1] <= ema_values[i - 1]:
            found_idx = i
            break

    if found_idx is None:
        return None

    entry = closes[found_idx]
    current_price = closes[-1]

    if abs(entry - current_price) / current_price > 0.05:
        return None

    # Adaptive R: Simple uses 3R in strong trends
    trend = compute_trend_strength(ema_values)
    r_mult = 3.0 if trend >= 2.0 else 2.0

    atr = compute_atr(candles)
    atr_result = build_atr_stop_and_target(entry, atr, r_mult)
    if atr_result is None:
        return None

    stop_loss, take_profit, risk_per_share = atr_result

    # Capital-based position sizing
    shares = floor(investment_dollars / entry)
    if shares <= 0:
        return None

    total_risk = shares * risk_per_share
    total_profit = shares * (take_profit - entry)

    return StrategyResult(
        strategy="simple",
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        shares=shares,
        risk_per_share=round(risk_per_share, 2),
        total_risk=round(total_risk, 2),
        total_profit=round(total_profit, 2),
        notes="Price closed above the 8 EMA after being below it. Stop and target sized using ATR; R-multiple adapts to trend strength."
    )


def swing_high_breakout(candles, ema_values, investment_dollars):
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    LOOKBACK_LIMIT = 10
    last_idx = len(candles) - 1
    start_idx = max(3, last_idx - LOOKBACK_LIMIT)

    swing_high_idx = None
    for i in range(start_idx, 2, -1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            if ema_values[i] is not None and closes[i] > ema_values[i]:
                swing_high_idx = i
                break

    if swing_high_idx is None:
        return None

    swing_high = highs[swing_high_idx]

    if closes[last_idx] <= swing_high:
        return None
    if ema_values[last_idx] is None or closes[last_idx] <= ema_values[last_idx]:
        return None

    entry = closes[last_idx]
    current_price = closes[-1]

    if abs(entry - current_price) / current_price > 0.05:
        return None

    # Adaptive R: Swing uses 3R when volatility is compressed
    vol = compute_volatility_compression(candles)
    r_mult = 3.0 if vol >= 1.0 else 2.0

    atr = compute_atr(candles)
    atr_result = build_atr_stop_and_target(entry, atr, r_mult)
    if atr_result is None:
        return None

    stop_loss, take_profit, risk_per_share = atr_result

    # Capital-based position sizing
    shares = floor(investment_dollars / entry)
    if shares <= 0:
        return None

    total_risk = shares * risk_per_share
    total_profit = shares * (take_profit - entry)

    return StrategyResult(
        strategy="swing",
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        shares=shares,
        risk_per_share=round(risk_per_share, 2),
        total_risk=round(total_risk, 2),
        total_profit=round(total_profit, 2),
        notes="Breakout above a recent swing high while price is above the 8 EMA. Stop and target sized using ATR; R-multiple adapts to volatility compression."
    )


def retest_breakout(candles, ema_values, investment_dollars):
    closes = [c["close"] for c in candles]
    LOOKBACK_LIMIT = 7

    last_idx = len(closes) - 1
    start_idx = max(2, last_idx - LOOKBACK_LIMIT)

    found_idx = None
    for i in range(last_idx, start_idx - 1, -1):
        if (
            ema_values[i] is None
            or ema_values[i - 1] is None
            or ema_values[i - 2] is None
        ):
            continue

        above_two_ago = closes[i - 2] > ema_values[i - 2]
        near_prev = closes[i - 1] <= ema_values[i - 1] * 1.01
        bounce = closes[i] > ema_values[i]

        if above_two_ago and near_prev and bounce:
            found_idx = i
            break

    if found_idx is None:
        return None

    entry = closes[found_idx]
    current_price = closes[-1]

    if abs(entry - current_price) / current_price > 0.05:
        return None

    # Adaptive R: Retest uses 3R only on perfect pullbacks
    pull = compute_pullback_quality(candles, ema_values)
    r_mult = 3.0 if pull >= 3.0 else 2.0

    atr = compute_atr(candles)
    atr_result = build_atr_stop_and_target(entry, atr, r_mult)
    if atr_result is None:
        return None

    stop_loss, take_profit, risk_per_share = atr_result

    # Capital-based position sizing
    shares = floor(investment_dollars / entry)
    if shares <= 0:
        return None

    total_risk = shares * risk_per_share
    total_profit = shares * (take_profit - entry)

    return StrategyResult(
        strategy="retest",
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        shares=shares,
        risk_per_share=round(risk_per_share, 2),
        total_risk=round(total_risk, 2),
        total_profit=round(total_profit, 2),
        notes="Price pulled back to the 8 EMA and then bounced back above it. Stop and target sized using ATR; R-multiple adapts to pullback quality."
    )


# -----------------------------
# API endpoint
# -----------------------------

@app.get("/strategy", response_model=StrategyResponse)
def get_strategy(
    ticker: str = Query(..., description="Stock ticker symbol, e.g., AAPL"),
    dollars: float = Query(..., gt=0, description="Capital to allocate to this trade in dollars"),
    type: str = Query("simple", description="Strategy type: simple, swing, retest, or all"),
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
            detail="No valid strategy signals found for the given inputs.",
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
