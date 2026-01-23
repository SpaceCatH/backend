# services/firestore_ops.py

from typing import List, Dict
from firebase_client import get_db


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

def normalize_ticker(ticker: str) -> str:
    """
    Firestore document IDs cannot contain periods.
    BRK.B → BRK-B
    """
    return ticker.replace(".", "-")


# ---------------------------------------------------------
# WRITE OPERATIONS
# ---------------------------------------------------------

def write_daily_scan_result(
    db,
    ticker: str,
    best,
    simple_res,
    swing_res,
    retest_res,
):
    """
    Writes a clean, validated daily scan document.
    Ensures no partial or malformed data is stored.
    """

    doc_id = normalize_ticker(ticker)

    payload = {
        "ticker": ticker,
        "best_strategy": best.strategy,
        "best_score": float(round(best.score, 2)),
        "has_simple": simple_res is not None,
        "has_swing": swing_res is not None,
        "has_retest": retest_res is not None,
        "updated_at": int(__import__("time").time()),
    }

    db.collection("daily_scan").document(doc_id).set(payload)


def delete_daily_scan_doc(db, doc_id: str):
    """
    Removes a ticker from the daily scan results.
    Used when no valid strategies exist.
    """
    db.collection("daily_scan").document(doc_id).delete()


# ---------------------------------------------------------
# READ OPERATIONS
# ---------------------------------------------------------

def read_daily_scan_candidates(db) -> List[Dict]:
    """
    Reads all daily scan documents from Firestore.
    Returns a list of raw dicts (not Pydantic models).
    """

    docs = db.collection("daily_scan").stream()
    results = []

    for doc in docs:
        data = doc.to_dict() or {}

        # Defensive filtering — skip empty or malformed docs
        if not isinstance(data, dict):
            continue

        # Ensure required fields exist
        if "ticker" not in data or "best_strategy" not in data or "best_score" not in data:
            continue

        results.append(data)

    return results