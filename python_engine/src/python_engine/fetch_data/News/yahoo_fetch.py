"""Utilities to fetch and prepare news articles for a ticker.

Provides light wrappers around yfinance news metadata and scraping helpers
used by the sentiment pipeline.
"""

import yfinance as yf
import pandas as pd
import trafilatura
import numpy as np
from src.python_engine.fetch_data.News.finBERT_process import get_day_score
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List


@dataclass
class NewsArticle:
    title: str
    publisher: str
    link: str
    pub_date: datetime
    summary: str


class NewsCollection:
    def __init__(self, ticker: str, articles: List[NewsArticle]):
        self.ticker = ticker
        self.articles = articles
        self.count = len(articles)


def fetch_ticker_news(ticker_symbol: str, days: int = 1) -> NewsCollection:
    """Fetch recent news metadata for `ticker_symbol` within the past `days` days.

    Args:
        ticker_symbol: Stock ticker to query via yfinance.
        days: Lookback window in days to include recent items.

    Returns:
        NewsCollection: Container with `articles` list and `count`.
    """
    ticker = yf.Ticker(ticker_symbol)
    raw_news = ticker.news
    
    cutoff_date = datetime.now() - timedelta(days=days)
    processed_articles = []

    for item in raw_news:
        # yfinance provides epoch seconds in 'providerPublishTime'
        pub_time = datetime.fromtimestamp(item.get('providerPublishTime', 0))
        
        if pub_time >= cutoff_date:
            article = NewsArticle(
                title=item.get('title', 'No Title'),
                publisher=item.get('publisher', 'Unknown'),
                link=item.get('link', ''),
                pub_date=pub_time,
                # Note: 'summary' is often a short snippet in the Yahoo API
                summary=item.get('summary', 'No summary available.')
            )
            processed_articles.append(article)

    return NewsCollection(ticker_symbol, processed_articles)

def get_all_urls(news_collection) -> List[str]:
    """Extract article URL strings from a `NewsCollection`.

    Args:
        news_collection: NewsCollection instance returned by `fetch_ticker_news`.

    Returns:
        List[str]: List of article URLs (may contain empty strings).
    """
    return [article.link for article in news_collection.articles]

def fetch_article_text(url: str) -> str:
    """Fetch full article text using `trafilatura`.

    Args:
        url: Article URL to download and extract.

    Returns:
        str: Extracted article text, or empty string if extraction failed.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text if text else ""
    except Exception:
        # Swallow network/parsing errors; caller handles empty text.
        pass
    return ""

def process_sentiment(texts: List[str], sentiment_func) -> float:
    """Compute mean sentiment over a list of article texts using `sentiment_func`.

    Args:
        texts: List of article text strings.
        sentiment_func: Callable that maps text -> numeric sentiment score.

    Returns:
        float: Mean sentiment score across processed articles (0.0 if none).
    """
    if not texts:
        return 0.0
    
    scores = []
    for text in texts:
        if text.strip(): # Only process if there is actual content
            try:
                # We assume import_func returns a float/int
                score = sentiment_func(text)
                scores.append(score)
            except Exception:
                continue
    
    return float(np.mean(scores)) if scores else 0.0

def news_inference_pipeline(ticker: str, days: int) -> tuple:
    """
    Complete pipeline: Fetch -> Scrape -> Sentiment.
    Returns: (sentiment_score, article_count)
    """
    # Documented inline: returns (float_score, int_count)
    try:
        # 1. Fetch news metadata (titles/links)
        news_collection = fetch_ticker_news(ticker, days=days)
        
        if not news_collection or news_collection.count == 0:
            return 0.0, 0

        # 2. Scrape full article text from URLs
        urls = get_all_urls(news_collection)
        full_texts = []
        
        for url in urls:
            content = fetch_article_text(url)
            if content:
                full_texts.append(content)

        # 3. Compute sentiment score (may be 0.0 if scraping failed)
        sentiment_score = process_sentiment(full_texts, get_day_score)
        article_count = news_collection.count

        return sentiment_score, article_count

    except Exception as e:
        # Global safety net for the pipeline
        print(f"Pipeline encountered an issue: {e}. Defaulting to neutral values.")
        return 0.0, 0