# utils/ema.py

from fastapi import HTTPException
from typing import List


def calculate_ema(prices: List[float], period: int = 8) -> List[float]:
    """
    Computes an EMA for the given list of prices.
    Returns a list where the first (period - 1) values are None,
    followed by the EMA series.
    """

    if len(prices) < period:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data to calculate EMA (need {period}, got {len(prices)})"
        )

    ema_values = []
    k = 2 / (period + 1)

    # Initial SMA for the first EMA value
    sma = sum(prices[:period]) / period
    ema_values.extend([None] * (period - 1))
    ema_values.append(sma)

    # Continue EMA calculation
    for price in prices[period:]:
        prev_ema = ema_values[-1]
        current_ema = (price - prev_ema) * k + prev_ema
        ema_values.append(current_ema)

    return ema_values