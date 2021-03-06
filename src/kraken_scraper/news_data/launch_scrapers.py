"""
launch_scrapers.py -> launches threaded news site scrapers. Currently only functional with
news_api websites but should serve as an interface reference for any other scrapers we
may create

Author: IanQS
"""

from news_api_key import key
from newsapi import NewsApiClient
from kraken_scraper.news_data.news_api_scraper import NewsAPIScraper
from kraken_scraper.news_data.site_configs import SITE_CONF

import threading

if __name__ == '__main__':
    query_kws = ['bitcoin']
    # First get all configured newsapi sites
    valid_names = set()
    for src_name in SITE_CONF.keys():
        if SITE_CONF[src_name]['api']:  # Only extract where we use newsAPI
            valid_names.add(src_name)

    # Then, launch those sites
    all_threads = []
    for src in valid_names:
        news_thread = threading.Thread(
            target=NewsAPIScraper.spawn, args=(src, query_kws), daemon=True
        )
        all_threads.append(news_thread)
        news_thread.start()

    # Launch all the non-news api compatible sites

    # Keeps main connection alive until all connections are "done"
    for news_thread in all_threads:
        news_thread.join()
