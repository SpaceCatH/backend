# utils/atr.py

from typing import List, Optional


# ---------------------------------------------------------
# ATR CALCULATION
# ---------------------------------------------------------

def compute_atr(candles: List[dict], period: int = 14) -> Optional[float]:
    """
    Computes the Average True Range (ATR) over the last `period` candles.
    Returns None if insufficient data.
    """

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
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        trs.append(tr)

    if len(trs) < period:
        return None

    recent_trs = trs[-period:]
    atr = sum(recent_trs) / period
    return atr


# ---------------------------------------------------------
# ATR-BASED STOP & TARGET
# ---------------------------------------------------------

MAX_STOP_PCT = 0.10       # Max 10% stop
ATR_MULT_STOP = 1.5       # 1.5 × ATR stop sizing


def build_atr_stop_and_target(entry: float, atr: Optional[float], r_mult: float):
    """
    Builds stop-loss and take-profit levels using ATR.
    Returns (stop_loss, take_profit, risk_per_share) or None.
    """

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


# ---------------------------------------------------------
# TREND STRENGTH
# ---------------------------------------------------------

def compute_trend_strength(ema_values: List[float]) -> float:
    """
    Measures trend strength based on EMA slope over last ~20 periods.
    Returns a score between 0 and 3.
    """

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


# ---------------------------------------------------------
# VOLATILITY COMPRESSION
# ---------------------------------------------------------

def compute_volatility_compression(candles: List[dict]) -> float:
    """
    Measures volatility compression by comparing current ATR to recent ATR.
    Returns a score between 0 and 2.
    """

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
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
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


# ---------------------------------------------------------
# PULLBACK QUALITY
# ---------------------------------------------------------

def compute_pullback_quality(candles: List[dict], ema_values: List[float]) -> float:
    """
    Measures the quality of a pullback relative to the 8 EMA.
    Returns a score between 0 and 3.
    """

    closes = [c["close"] for c in candles]
    if len(closes) < 3 or len(ema_values) < 3:
        return 0.0

    i = len(closes) - 1

    if (
        ema_values[i] is None
        or ema_values[i - 1] is None
        or ema_values[i - 2] is None
    ):
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