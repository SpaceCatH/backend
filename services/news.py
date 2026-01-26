# services/news.py

import os
import httpx
from models.strategy import NewsItem

# Load your FMP API key from environment variables
FMP_API_KEY = os.getenv("FMP_API_KEY")

async def fetch_recent_news(ticker: str):
    if not FMP_API_KEY:
        # Graceful fallback if key is missing
        return []

    url = (
        f"https://financialmodelingprep.com/api/v3/stock_news"
        f"?tickers={ticker}&limit=5&apikey={FMP_API_KEY}"
    )

    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        data = r.json()

    news_items = []
    for item in data:
        news_items.append(
            NewsItem(
                title=item.get("title", ""),
                source=item.get("site", ""),
                published_at=item.get("publishedDate", ""),
                url=item.get("url", "")
            )
        )

    return news_items
