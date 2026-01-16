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
    """
    Fetches news for a ticker and filters by the last 'n' days.
    """
    ticker = yf.Ticker(ticker_symbol)
    raw_news = ticker.news
    
    cutoff_date = datetime.now() - timedelta(days=days)
    processed_articles = []

    for item in raw_news:
        # yfinance provides time in UTC seconds
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
    """Extracts all links from a NewsCollection object."""
    return [article.link for article in news_collection.articles]

def fetch_article_text(url: str) -> str:
    """Fetches full text from a URL using trafilatura with a fallback."""
    try:
        # Trafilatura is excellent at stripping ads/menus to get just the body
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text if text else ""
    except Exception:
        # If primary fetch fails, we avoid exceptions and return empty
        pass
    return ""

def process_sentiment(texts: List[str], sentiment_func) -> float:
    """Computes average sentiment for a list of strings."""
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
    try:
        # 1. Fetch news summaries/links
        # (Assuming the fetch_ticker_news function from the previous step)
        news_collection = fetch_ticker_news(ticker, days=days)
        
        if not news_collection or news_collection.count == 0:
            return 0.0, 0

        # 2. Get URLs and scrape full text
        urls = get_all_urls(news_collection)
        full_texts = []
        
        for url in urls:
            content = fetch_article_text(url)
            if content:
                full_texts.append(content)

        # 3. Analyze Sentiment
        # If scraping failed for all but news count is still > 0, 
        # we still have a volume but 0 sentiment.
        sentiment_score = process_sentiment(full_texts, get_day_score)
        article_count = news_collection.count

        return sentiment_score, article_count

    except Exception as e:
        # Global safety net for the pipeline
        print(f"Pipeline encountered an issue: {e}. Defaulting to neutral values.")
        return 0.0, 0