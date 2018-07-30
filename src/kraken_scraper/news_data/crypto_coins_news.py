from .base_scraper import GenericScraper
from news_api_key import key
from newsapi import NewsApiClient

class CryptoCoinNews(GenericScraper):
    def __init__(self):
        api = NewsApiClient(api_key=key)
        super().__init__('/crypto_coins_news', )

    def _save_article(seen_article):

        super().save_article(article, url)


    def get_articles():
        while True:



if __name__ == '__main__':
    scraper = CryptoCoinNews()
    scraper.
