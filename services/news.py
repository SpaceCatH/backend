# services/news.py

import os
import httpx
from models.strategy import NewsItem

FMP_API_KEY = os.getenv("FMP_API_KEY")

async def fetch_recent_news(ticker: str):
    if not FMP_API_KEY:
        return []

    url = (
    f"https://financialmodelingprep.com/api/v4/news"
    f"?tickers={ticker}&limit=5&apikey={FMP_API_KEY}"
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
    
