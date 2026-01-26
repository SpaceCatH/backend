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

    
    if not isinstance(data, list):
        return []

    news_items = []
    for item in data:
        if not isinstance(item, dict):
            continue  # skip strings or malformed entries

        published = item.get("publishedDate", "")

        if " " in published:
            date_part, time_part = published.split(" ", 1)
        else:
            date_part, time_part = published, ""

        news_items.append(
            NewsItem(
                title=item.get("title", ""),
                source=item.get("site", ""),
                date=date_part,
                time=time_part,
                url=item.get("url", "")
            )
        )

    return news_items
    
