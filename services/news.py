# services/news.py

import os
import httpx
from datetime import datetime, timezone

NEWS_API_KEY = os.getenv("abbbd8715e7d428cbcd8628d952df1bb")

# You can swap this URL later without changing the rest of the code
NEWS_API_URL = "https://newsapi.org/v2/everything"


async def fetch_recent_news(ticker: str, limit: int = 3):
    """
    Fetches recent news articles for a ticker.
    Returns a list of dicts with:
    - title
    - source
    - date (YYYY-MM-DD)
    - time (HH:MM, local)
    - url
    """

    if not NEWS_API_KEY:
        # Fail silently but safely
        return []

    params = {
        "q": ticker,
        "sortBy": "publishedAt",
        "pageSize": limit,
        "language": "en",
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                NEWS_API_URL,
                params=params,
                headers={"Authorization": f"Bearer {NEWS_API_KEY}"},
            )
            resp.raise_for_status()
    except Exception:
        return []

    data = resp.json()
    articles = data.get("articles", [])

    results = []

    for a in articles:
        published = a.get("publishedAt")  # e.g. "2026-01-25T14:32:00Z"
        date_str = ""
        time_str = ""

        if published:
            try:
                # Convert to datetime
                dt = datetime.fromisoformat(published.replace("Z", "+00:00"))

                # Convert to local timezone
                dt_local = dt.astimezone()

                date_str = dt_local.date().isoformat()
                time_str = dt_local.strftime("%H:%M")
            except Exception:
                pass

        results.append(
            {
                "title": a.get("title", ""),
                "source": (a.get("source") or {}).get("name", ""),
                "date": date_str,
                "time": time_str,
                "url": a.get("url", ""),
            }
        )

    return results
