# models/scan.py

from pydantic import BaseModel
from typing import List


class ScanCandidate(BaseModel):
    ticker: str
    best_strategy: str
    best_score: float
    has_simple: bool
    has_swing: bool
    has_retest: bool


class ScanResponse(BaseModel):
    candidates: List[ScanCandidate]