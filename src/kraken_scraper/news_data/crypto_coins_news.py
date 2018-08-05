"""
crypto_coins_news - scraper for aforementioned news site.

Author: Ian Q.

Notes:

"""
import sys

from .base_scraper import GenericScraper
from .site_configs import SITE_CONF
from .news_api_key import key

from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
 
class CryptoCoinNews(GenericScraper):
    def __init__(self, source):
        self.api = NewsApiClient(api_key=key)
        self.source = source
        self.config = SITE_CONF[source]
        super().__init__('/{}'.format(self.source))
        self.get_articles()

    def _process(self, query) -> dict:
        if self.config['selenium']:
            chrome_options = Options()
            chrome_options.add_argument("--headless")

            driver = webdriver.Chrome(chrome_options=chrome_options)
            driver.get(query['url'])
            soup = BeautifulSoup(driver.page_source, "html.parser")
        else:
            soup = BeautifulSoup(requests.get(query['url']).text, "html.parser")

        query["article"] = str(soup.find(self.config["html_tag"], self.config["tag_attributes"]))
        if query["article"] == "None": # query['article'] is None when content not found, log this error
            # TODO: log this error somewhere
            print("Error: Article not found for {}".format(query['url']))

        holder = {}
        for access_key in ['article', 'url', 'date', 'site', 'author']:
            holder[access_key] = query[access_key]
        return holder

    def process(self, query_results: list) -> None:
        """ Process the queries - do any scraping on them
        before calling self.save_article()

        :param queries:
        :return:
            None
        """
        for query in query_results["articles"]:
            query = self.__substitution(query)
            processed_data = self._process(query)
            self.save_article(**processed_data)

    def __substitution(self, query):
        """ Fixes query to have the fields expected by our saver

        :param query:
        :return:
        """
        query['site'] = query['source']['name']
        query['date'] = query['publishedAt']
        return query


    def get_articles(self):
        """ Infinite while-loop that queries the website for any
        articles it has not seen

        :return:
        """
        ind = 1  # NewsAPIClient starts indexing at 1
        page_size = 100
        while True:
            queries, ind = self.safe_query(ind, page_size=page_size)
            if queries is None:
                page_size=10
            self.process(queries)

    def safe_query(self, ind, page_size=100):
        error_message = 'Error retrieving from {}. Error: {}'
        try:
            if ind is None:  # Only here if we've hit maximumResultsReached before
                ind = 1
            q = self.api.get_everything(q='bitcoin',
                                        sources=self.source,
                                        language='en',
                                        page_size=page_size,
                                        sort_by='publishedAt',
                                        page=ind)
        except NewsAPIException as e:
            ################################################
            # Reached end of queries. Reset to first page
            ################################################
            if e['code'] == 'maximumResultsReached':
                return None, None
        except Exception as e:
            print(error_message.format(self.__class__.__name__, e))
            sys.exit(0)
        if q['status'] != 'ok':
            print(error_message.format(self.__class__.__name__, 'Status: {}'.format(q['status'])))
        return q, ind + 1

if __name__ == '__main__':
    scraper = CryptoCoinNews('crypto-coins-news')
