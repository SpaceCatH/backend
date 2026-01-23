# services/strategies.py

from math import floor
from typing import Optional, List

from models.strategy import StrategyResult
from utils.atr import (
    compute_atr,
    build_atr_stop_and_target,
    compute_trend_strength,
    compute_volatility_compression,
    compute_pullback_quality,
)


# ---------------------------------------------------------
# SIMPLE EMA BREAKOUT
# ---------------------------------------------------------

def simple_ema_breakout(candles, ema_values, investment_dollars) -> Optional[StrategyResult]:
    closes = [c["close"] for c in candles]
    if len(closes) < 3 or len(ema_values) < 3:
        return None

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
    if entry <= 0:
        return None

    # Avoid stale entries
    if abs(entry - current_price) / current_price > 0.05:
        return None

    trend = compute_trend_strength(ema_values)
    r_mult = 3.0 if trend >= 2.0 else 2.0

    atr = compute_atr(candles)
    atr_result = build_atr_stop_and_target(entry, atr, r_mult)
    if atr_result is None:
        return None

    stop_loss, take_profit, risk_per_share = atr_result

    shares = floor(investment_dollars / entry)
    if shares <= 0:
        return None

    total_risk = shares * risk_per_share
    total_profit = shares * (take_profit - entry)

    if total_risk <= 0 or total_profit <= 0:
        return None

    return StrategyResult(
        strategy="simple",
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        shares=shares,
        risk_per_share=round(risk_per_share, 2),
        total_risk=round(total_risk, 2),
        total_profit=round(total_profit, 2),
        notes="Price closed above the 8 EMA after being below it. ATR-based stop and adaptive R-multiple."
    )


# ---------------------------------------------------------
# SWING HIGH BREAKOUT
# ---------------------------------------------------------

def swing_high_breakout(candles, ema_values, investment_dollars) -> Optional[StrategyResult]:
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]

    if len(candles) < 5 or len(ema_values) < 5:
        return None

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
    if entry <= 0:
        return None

    if abs(entry - current_price) / current_price > 0.05:
        return None

    vol = compute_volatility_compression(candles)
    r_mult = 3.0 if vol >= 1.0 else 2.0

    atr = compute_atr(candles)
    atr_result = build_atr_stop_and_target(entry, atr, r_mult)
    if atr_result is None:
        return None

    stop_loss, take_profit, risk_per_share = atr_result

    shares = floor(investment_dollars / entry)
    if shares <= 0:
        return None

    total_risk = shares * risk_per_share
    total_profit = shares * (take_profit - entry)

    if total_risk <= 0 or total_profit <= 0:
        return None

    return StrategyResult(
        strategy="swing",
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        shares=shares,
        risk_per_share=round(risk_per_share, 2),
        total_risk=round(total_risk, 2),
        total_profit=round(total_profit, 2),
        notes="Breakout above a recent swing high while above the 8 EMA. ATR-based stop and volatility-adjusted R-multiple."
    )


# ---------------------------------------------------------
# RETEST BREAKOUT
# ---------------------------------------------------------

def retest_breakout(candles, ema_values, investment_dollars) -> Optional[StrategyResult]:
    closes = [c["close"] for c in candles]
    if len(closes) < 3 or len(ema_values) < 3:
        return None

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
    if entry <= 0:
        return None

    if abs(entry - current_price) / current_price > 0.05:
        return None

    pull = compute_pullback_quality(candles, ema_values)
    r_mult = 3.0 if pull >= 3.0 else 2.0

    atr = compute_atr(candles)
    atr_result = build_atr_stop_and_target(entry, atr, r_mult)
    if atr_result is None:
        return None

    stop_loss, take_profit, risk_per_share = atr_result

    shares = floor(investment_dollars / entry)
    if shares <= 0:
        return None

    total_risk = shares * risk_per_share
    total_profit = shares * (take_profit - entry)

    if total_risk <= 0 or total_profit <= 0:
        return None

    return StrategyResult(
        strategy="retest",
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        shares=shares,
        risk_per_share=round(risk_per_share, 2),
        total_risk=round(total_risk, 2),
        total_profit=round(total_profit, 2),
        notes="Pullback to the 8 EMA followed by a bounce. ATR-based stop and pullback-quality-adjusted R-multiple."
    )