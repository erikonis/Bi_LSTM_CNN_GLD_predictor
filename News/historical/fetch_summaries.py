from datetime import date
import re
import pandas as pd
import trafilatura, requests
from concurrent.futures import ThreadPoolExecutor
import time, random
import csv, json, os

def get_wayback_url(original_url):
    """Checks the Wayback Machine API for the closest archived snapshot."""
    api_url = f"http://archive.org/wayback/available?url={original_url}"
    try:
        response = requests.get(api_url, timeout=10)
        data = response.json()
        if data.get("archived_snapshots"):
            return data["archived_snapshots"]["closest"]["url"]
    except Exception:
        return None
    return None

def process_yahoo_url(url):
    """
    Downloads and extracts the clean text from a Yahoo Finance URL.
    """

    time.sleep(random.uniform(1, 3.5))
    
    try:
        # 1. Fetch the page with a browser-like header to avoid 403 Forbidden
        downloaded = trafilatura.fetch_url(url)
        source_type = "live"
        
        # If fails, we try archives
        if downloaded is None:
            print(f"Live URL failed, checking Wayback: {url}")
            archive_url = get_wayback_url(url)
            if archive_url:
                downloaded = trafilatura.fetch_url(archive_url)

                if downloaded is not None:
                    url = archive_url
                    source_type = "archived"
                else:
                    return {"url": url, "text": None, "status": "failed"}

        # 2. Extract content (removes headers, footers, and ads)
        # 'include_comments=False' keeps the data clean for FinBERT
        content = trafilatura.extract(downloaded, include_comments=False, no_fallback=False)
        
        if content:
            # FinBERT usually takes the first 512 tokens. 
            # We save the first 1000 characters as the 'Summary'
            summary = content[:1000].replace('\n', ' ')
            return {"url": url, "summary": summary, "full_text": content, "status": "success", "source_type": source_type}
        
    except Exception as e:
        return {"url": url, "text": str(e), "status": "error"}
    
    return {"url": url, "text": None, "status": "empty"}
   
def process_yahoo_urls(urls : dict, batch_size : int, output_dir : str, output_name : str):
    results = {}

    for i in range(0, len(urls), batch_size):
        batch = list(urls.values())[i:i+batch_size]

        # Use ThreadPool to process multiple URLs at once (speeds up 10 years of data)
        with ThreadPoolExecutor(max_workers=5) as executor:
            out = list(executor.map(process_yahoo_url, batch))
        
        for j, date in enumerate(list(urls.keys())[i:i+batch_size]):
            results[date] = out[j]
        #Cooldown
        time.sleep(10)

    os.makedirs(output_dir, exist_ok=True)
    json.dump(results, open(f"{output_dir}/{output_name}.json", "w"), indent=4)

def csv_to_dict_url(path : str, date_name : str = "DATE", url_name : str = "page_url"):
    """
    Turning the entries from csv file into a list. By default the code looks for a 
    header named 'url' and fetches them into a list.
    
    :param path: Path to the csv file to read.
    :type path: str
    :param date_name: How in the csv file the date column is named.
    :type date_name: str
    :param url_name: How in the csv file the url column is named.
    :type url_name: str
    """
    urls = {}
    with open(path, "r") as file:
        rows = csv.reader(file)
    
        # Getting the column of urls
        header = next(rows)
        #print(header)
        url_idx = header.index(url_name)
        date_idx = header.index(date_name)

        for row in rows:
            urls[row[date_idx]] = row[url_idx]
    return urls

if __name__ == "__main__":
    for option in ["2021-2022", "2022-2023", "2023-2024", "2024-2025"]:
        print(process_yahoo_urls(csv_to_dict_url(f"csvs-url/{option}.csv"), output_name=option, batch_size=50, output_dir="summaries"))