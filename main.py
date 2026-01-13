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

# Allow frontend (React) to call this API from another origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this later
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
    notes: str


class StrategyResponse(BaseModel):
    strategies: List[StrategyResult]


# -----------------------------
# Helper functions
# -----------------------------

def fetch_eod_data(ticker: str, limit: int = 120):
    """
    Fetch end-of-day data (daily candles) from Alpha Vantage.
    Returns a list of candles sorted from oldest to newest:
    [
        {"date": "YYYY-MM-DD", "open": ..., "high": ..., "low": ..., "close": ...},
        ...
    ]
    """
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

    # Sort by date ascending (oldest -> newest)
    candles.sort(key=lambda x: x["date"])
    return candles[-limit:]


def calculate_ema(prices: List[float], period: int = 8) -> List[float]:
    """
    Calculate EMA for a list of closing prices.
    Returns a list of EMA values of the same length as prices.
    """
    if len(prices) < period:
        raise HTTPException(status_code=400, detail="Not enough data to calculate EMA")

    ema_values = []
    k = 2 / (period + 1)

    # Start with simple average for the first EMA
    sma = sum(prices[:period]) / period
    ema_values.extend([None] * (period - 1))  # padding for alignment
    ema_values.append(sma)

    # Continue with EMA formula
    for price in prices[period:]:
        prev_ema = ema_values[-1]
        current_ema = (price - prev_ema) * k + prev_ema
        ema_values.append(current_ema)

    return ema_values


def compute_position_size(investment_dollars: float, entry: float, stop_loss: float) -> (int, float, float):
    """
    Compute the number of shares based on a basic risk rule:
    - Risk 1% of total investment per trade.
    """
    if entry <= 0 or stop_loss <= 0:
        return 0, 0.0, 0.0

    risk_per_trade = investment_dollars * 0.01  # 1% risk
    risk_per_share = abs(entry - stop_loss)

    if risk_per_share == 0:
        return 0, 0.0, 0.0

    shares = floor(risk_per_trade / risk_per_share)
    total_risk = shares * risk_per_share

    return shares, risk_per_share, total_risk


# -----------------------------
# Strategy implementations
# -----------------------------

def simple_ema_breakout(candles, ema_values, investment_dollars) -> Optional[StrategyResult]:
    """
    Simple EMA breakout:
    - Entry: last close where price closes above the 8 EMA after being below.
    - Stop: a bit below the EMA (using the last EMA value).
    - Take profit: 2R (twice the risk).
    """

    closes = [c["close"] for c in candles]

    # Find the most recent bar where close crosses above EMA
    idx = len(closes) - 1
    found_idx = None
    for i in range(len(closes) - 1, 0, -1):
        if ema_values[i] is None or ema_values[i - 1] is None:
            continue
        if closes[i] > ema_values[i] and closes[i - 1] <= ema_values[i - 1]:
            found_idx = i
            break

    if found_idx is None:
        return None

    entry = closes[found_idx]
    ema_at_entry = ema_values[found_idx]
    stop_loss = ema_at_entry * 0.99  # 1% below EMA as a simple buffer
    # Risk per share:
    risk_per_share = entry - stop_loss
    take_profit = entry + 2 * risk_per_share

    shares, rps, total_risk = compute_position_size(investment_dollars, entry, stop_loss)

    if shares <= 0:
        return None

    return StrategyResult(
        strategy="simple",
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        shares=shares,
        risk_per_share=round(rps, 2),
        total_risk=round(total_risk, 2),
        notes="Price closed above the 8 EMA after being below it."
    )


def swing_high_breakout(candles, ema_values, investment_dollars) -> Optional[StrategyResult]:
    """
    Swing-high breakout:
    - Identify a recent swing high.
    - Entry: breakout above that swing high while price is above EMA.
    - Stop: below the swing low.
    - Take profit: 1.5R.
    """

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    # Define a simple swing high: high[i] > high[i-1] and high[i] > high[i+1]
    swing_high_idx = None
    for i in range(len(highs) - 3, 2, -1):  # search backwards
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            # Ensure price is above EMA around this region
            if ema_values[i] is not None and closes[i] > ema_values[i]:
                swing_high_idx = i
                break

    if swing_high_idx is None:
        return None

    swing_high = highs[swing_high_idx]

    # Assume breakout happens at the last candle closing above the swing high
    last_idx = len(candles) - 1
    if closes[last_idx] <= swing_high or ema_values[last_idx] is None or closes[last_idx] <= ema_values[last_idx]:
        return None

    entry = closes[last_idx]

    # Simple swing low: minimum low over a small window before the swing high
    window_start = max(0, swing_high_idx - 5)
    swing_low = min(lows[window_start:swing_high_idx + 1])

    stop_loss = swing_low * 0.99  # small buffer below swing low
    risk_per_share = entry - stop_loss
    take_profit = entry + 1.5 * risk_per_share

    shares, rps, total_risk = compute_position_size(investment_dollars, entry, stop_loss)

    if shares <= 0:
        return None

    return StrategyResult(
        strategy="swing",
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        shares=shares,
        risk_per_share=round(rps, 2),
        total_risk=round(total_risk, 2),
        notes="Breakout above a recent swing high while price is above the 8 EMA."
    )


def retest_breakout(candles, ema_values, investment_dollars) -> Optional[StrategyResult]:
    """
    Retest breakout:
    - Price is above EMA.
    - Pulls back to the EMA (or slightly below).
    - Then closes back above EMA (bounce).
    - Entry: on the bounce close.
    - Stop: slightly below EMA.
    - Take profit: 2R.
    """

    closes = [c["close"] for c in candles]

    idx = len(closes) - 1
    found_idx = None
    for i in range(len(closes) - 1, 1, -1):
        if ema_values[i] is None or ema_values[i - 1] is None or ema_values[i - 2] is None:
            continue

        # Condition:
        # - 2 bars ago: clearly above EMA
        # - 1 bar ago: close near or slightly below EMA (retest)
        # - current bar: close back above EMA (bounce)
        above_two_ago = closes[i - 2] > ema_values[i - 2]
        near_or_below_prev = closes[i - 1] <= ema_values[i - 1] * 1.01
        bounce_now = closes[i] > ema_values[i]

        if above_two_ago and near_or_below_prev and bounce_now:
            found_idx = i
            break

    if found_idx is None:
        return None

    entry = closes[found_idx]
    ema_at_entry = ema_values[found_idx]
    stop_loss = ema_at_entry * 0.99
    risk_per_share = entry - stop_loss
    take_profit = entry + 2 * risk_per_share

    shares, rps, total_risk = compute_position_size(investment_dollars, entry, stop_loss)

    if shares <= 0:
        return None

    return StrategyResult(
        strategy="retest",
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        shares=shares,
        risk_per_share=round(rps, 2),
        total_risk=round(total_risk, 2),
        notes="Price pulled back to the 8 EMA and then bounced back above it."
    )


# -----------------------------
# API endpoint
# -----------------------------

@app.get("/strategy", response_model=StrategyResponse)
def get_strategy(
    ticker: str = Query(..., description="Stock ticker symbol, e.g., AAPL"),
    dollars: float = Query(..., gt=0, description="Investment amount in dollars"),
    type: str = Query("simple", description="Strategy type: simple, swing, retest, or all"),
):
    """
    Main endpoint to generate strategies.
    """

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
            detail="No valid strategy signals found for the given inputs. Try another ticker or date range.",
        )

    return StrategyResponse(strategies=strategies)


# -----------------------------
# For local / Replit run
# -----------------------------

