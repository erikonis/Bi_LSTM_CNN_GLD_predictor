from gdeltdoc import GdeltDoc, Filters
import pandas as pd
from datetime import datetime, timedelta

def fetch_gdelt_news(keyword: str, days_back: int = 90):
    """
    Fetch news from GDELT for a company/keyword.
    Note: GDELT API only goes back ~3 months at a time.
    """
    gd = GdeltDoc()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    f = Filters(
        keyword=keyword,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        country="US"  # Optional: filter by country
    )
    
    print(f"Fetching GDELT news for '{keyword}'")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Get articles
    articles = gd.article_search(f)
    
    print(f"Found {len(articles)} articles")
    return articles


def fetch_gdelt_timeline(keyword: str, days_back: int = 90):
    """Get timeline of article volume and tone."""
    gd = GdeltDoc()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    f = Filters(
        keyword=keyword,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    # Get tone timeline (sentiment proxy)
    tone_timeline = gd.timeline_search("timelinetone", f)
    
    # Get volume timeline
    volume_timeline = gd.timeline_search("timelinevolraw", f)
    
    return tone_timeline, volume_timeline


if __name__ == "__main__":
    # Search for Barrick Gold news
    # Use company name, not ticker (GDELT searches article text)
    articles = fetch_gdelt_news("Barrick Gold", days_back=90)
    
    if len(articles) > 0:
        print(f"\nColumns: {articles.columns.tolist()}")
        print(f"\nSample articles:")
        print(articles[['seendate', 'title', 'domain']].head(10))
        
        # Save to CSV
        articles.to_csv("barrick_gold_news.csv", index=False)
        print(f"\nSaved to barrick_gold_news.csv")
    
    # Also get sentiment timeline
    tone, volume = fetch_gdelt_timeline("Barrick Gold", days_back=90)
    print(f"\nTone timeline:")
    print(tone.head())
