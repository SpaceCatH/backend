# models/strategy.py

from pydantic import BaseModel
from typing import List


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