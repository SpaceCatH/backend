# services/news.py

import os
import httpx
from models.strategy import NewsItem

MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")

async def fetch_recent_news(ticker: str):
    if not MARKETAUX_API_KEY:
        return []

    url = (
        "https://api.marketaux.com/v1/news/all"
        f"?symbols={ticker}"
        "&filter_entities=true"
        "&language=en"
        "&limit=5"
        f"&api_token={MARKETAUX_API_KEY}"
    )

    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        data = r.json()

    articles = data.get("data", [])
    news_items = []

    for item in articles:
        published = item.get("published_at", "")

        if "T" in published:
            date_part, time_part = published.split("T", 1)
            time_part = time_part.replace("Z", "")
        else:
            date_part, time_part = published, ""

        news_items.append(
            NewsItem(
                title=item.get("title", ""),
                source=item.get("source", ""),
                date=date_part,
                time=time_part,
                url=item.get("url", "")
            )
        )

    return news_items
    
