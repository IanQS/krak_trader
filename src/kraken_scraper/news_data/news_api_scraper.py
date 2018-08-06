"""
news_api_scraper - scraper for aforementioned news site.

__main__ -> runs the scraper on a single site instance as a test.

Author: Ian Q.

Notes:

"""
import sys

from .base_scraper import GenericScraper
from .site_configs import SITE_CONF
from .news_api_key import key
from time import sleep as thread_swap # sleep to allow other threads to run

from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class NewsAPIScraper(GenericScraper):
    def __init__(self, source):
        self.api = NewsApiClient(api_key=key)
        self.source = source
        assert source in SITE_CONF.keys()  # Only scrape on websites we've configured
        super().__init__('/{}'.format(self.source))
        self.get_articles()

    @classmethod
    def spawn(cls, src_name):
        """ Class factory that spawns our instances

        src_name: valid news source name compatible with news_api
        """
        return cls(src_name)

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
                page_size=10  # We've hit an error, or reached end of all articles
            thread_swap(1)
            self.process(queries)

    def _fetch_website(self, query, sel_driver):
        if SITE_CONF[query['source']['id']]['selenium']:
            sel_driver.get(query['url'])
            soup = BeautifulSoup(sel_driver.page_source, "html.parser")
        else:
            soup = BeautifulSoup(requests.get(query['url']).text, "html.parser")

        return soup

    def _process(self, query, soup) -> dict:
        config = SITE_CONF[query['source']['id']]
        query["article"] = str(soup.find(config["html_tag"], config["tag_attributes"]))

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
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        sel_driver = webdriver.Chrome(chrome_options=chrome_options)

        for query in query_results["articles"]:
            query = self.__substitution(query)
            soup = self._fetch_website(query, sel_driver)
            processed_data = self._process(query, soup)
            self.save_article(**processed_data)
            thread_swap(0.1)

    def __substitution(self, query: dict):
        """ Fixes query to have the fields expected by our saver

        :param query:
        :return:
        """
        query['site'] = query['source']['name']
        query['date'] = query['publishedAt']
        return query

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
            print(error_message.format(self.name, e))
            sys.exit(0)
        if q['status'] != 'ok':
            print(error_message.format(self.name, 'Status: {}'.format(q['status'])))
        return q, ind + 1

if __name__ == '__main__':
    scraper = NewsAPIScraper('crypto-coins-news')  # Test name
