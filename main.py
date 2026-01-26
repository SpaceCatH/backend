# backend/main.py

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import time

from models.strategy import StrategyResponse, StrategyResult
from models.scan import ScanResponse, ScanCandidate
from services.alpha_vantage import fetch_eod_data
from services.strategies import (
    simple_ema_breakout,
    swing_high_breakout,
    retest_breakout,
)
from services.firestore_ops import (
    get_db,
    normalize_ticker,
    write_daily_scan_result,
    delete_daily_scan_doc,
    read_daily_scan_candidates,
)
from services.tickers import get_tickers_cached
from utils.ema import calculate_ema
from utils.atr import (
    compute_atr,
    compute_trend_strength,
    compute_volatility_compression,
    compute_pullback_quality,
)
from services.news import fetch_recent_news

# -----------------------------
# FastAPI app setup
# -----------------------------

app = FastAPI(
    title="8 EMA Swing Trading Strategy API",
    description="Generates entry/exit points for multiple EMA breakout strategies and scans the S&P 100.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Strategy endpoint
# -----------------------------

@app.get("/strategy", response_model=StrategyResponse)
async def get_strategy(
    ticker: str = Query(...),
    dollars: float = Query(..., gt=0),
    type: str = Query("simple"),
):

    print("STRATEGY REQUEST STARTED")

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
            detail="No valid strategy setups were found for this ticker.",
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

    recent_news = await fetch_recent_news(ticker_upper)

    return StrategyResponse(strategies=strategies,recent_news=recent_news)

# -----------------------------
# Daily scan endpoint (for Cloud Scheduler)
# -----------------------------

@app.post("/daily-scan")
def daily_scan(
    dollars: float = 10000,
    type: str = "all",
):
    """
    Runs the full S&P 100 scan, computes strategies, and writes one Firestore document per ticker.
    Intended to be triggered once per day by Cloud Scheduler.
    """
    strategy_type = type.lower()
    db = get_db()

    for idx, ticker in enumerate(SP100_TICKERS):
        ticker_upper = ticker.upper()
        print(f"Processing {ticker_upper}")
        try:
            candles = fetch_eod_data(ticker_upper)
            closes = [c["close"] for c in candles]
            ema_values = calculate_ema(closes, period=8)

            strategies: List[StrategyResult] = []

            simple_res = swing_res = retest_res = None

            if strategy_type in ("simple", "all"):
                simple_res = simple_ema_breakout(candles, ema_values, dollars)
                if simple_res is not None:
                    strategies.append(simple_res)

            if strategy_type in ("swing", "all"):
                swing_res = swing_high_breakout(candles, ema_values, dollars)
                if swing_res is not None:
                    strategies.append(swing_res)

            if strategy_type in ("retest", "all"):
                retest_res = retest_breakout(candles, ema_values, dollars)
                if retest_res is not None:
                    strategies.append(retest_res)

            if not strategies:
                doc_id = normalize_ticker(ticker_upper)
                delete_daily_scan_doc(db, doc_id)
                continue

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

            write_daily_scan_result(
                db=db,
                ticker=ticker_upper,
                best=best,
                simple_res=simple_res,
                swing_res=swing_res,
                retest_res=retest_res,
            )

        except HTTPException as e:
            print(f"HTTPException for {ticker_upper}: {e.detail}")
            continue
        except Exception as e:
            print(f"Error for {ticker_upper}: {e}")
            continue

        # Respect Alpha Vantage free tier: 5 calls per minute
        if (idx + 1) % 5 == 0 and (idx + 1) < len(SP100_TICKERS):
            time.sleep(70)

    return {"status": "completed"}

# -----------------------------
# Scan endpoint (reads from Firestore)
# -----------------------------

@app.get("/scan", response_model=ScanResponse)
def scan_sp100():
    """
    Returns the most recent daily scan results from Firestore.
    This is what the frontend should call for instant screener data.
    """
    db = get_db()
    raw_docs = read_daily_scan_candidates(db)

    candidates: List[ScanCandidate] = []
    for data in raw_docs:
        ticker = data.get("ticker")
        best_strategy = data.get("best_strategy")
        best_score = data.get("best_score")
        has_simple = data.get("has_simple")
        has_swing = data.get("has_swing")
        has_retest = data.get("has_retest")

        if (
            not ticker
            or not isinstance(best_strategy, str)
            or best_strategy == ""
            or best_score is None
        ):
            continue

        try:
            candidate = ScanCandidate(
                ticker=str(ticker),
                best_strategy=str(best_strategy),
                best_score=float(best_score),
                has_simple=True if has_simple is True else False,
                has_swing=True if has_swing is True else False,
                has_retest=True if has_retest is True else False,
            )
            candidates.append(candidate)
        except Exception as e:
            print("Skipping malformed scan doc:", data, "Error:", e)
            continue

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No cached scan results found in Firestore. Has /daily-scan run yet?",
        )

    candidates.sort(key=lambda c: c.best_score, reverse=True)
    top_candidates = candidates[:10]

    return ScanResponse(candidates=top_candidates)

# -----------------------------
# Tickers endpoint
# -----------------------------

@app.get("/tickers")
def get_tickers():
    return get_tickers_cached()
