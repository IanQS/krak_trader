"""
news_api_scraper - scraper for aforementioned news site.

__main__ -> runs the scraper on a single site instance as a test.

Author: Ian Q. and Bryan Q.

Edits:
    Bryan Q: added Selenium Scraper

Notes:

"""
import sys

from kraken_scraper.news_data.base_scraper import GenericScraper
from kraken_scraper.news_data.site_configs import SITE_CONF
from news_api_key import key
from time import sleep as thread_swap # sleep to allow other threads to run

from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

PAGE_LOAD_TIME = 10 # num of seconds the async request will wait for a page to load before giving up

class NewsAPIScraper(GenericScraper):
    def __init__(self, source, query_kws):
        self.api = NewsApiClient(api_key=key)
        self.source = source
        self.query_kws = query_kws
        assert source in SITE_CONF.keys()  # Only scrape on websites we've configured
        super().__init__('/{}'.format(self.source))
        self.config = SITE_CONF[source]
        self.driver = self.__setup_driver()
        self.get_articles()

    def __setup_driver(self):
        capabilities = DesiredCapabilities.CHROME
        capabilities["pageLoadStrategy"] = "none"
        chrome_options = Options()
        chrome_options.add_argument("--headless")

        driver = webdriver.Chrome(desired_capabilities=capabilities, chrome_options=chrome_options)
        async_wait_first = WebDriverWait(driver, PAGE_LOAD_TIME)
        async_wait_second = WebDriverWait(driver, PAGE_LOAD_TIME * 3)
        err_time = 'Selenium timed out after {} seconds for {} on {}'
        def wrapped_check(url):
            elem = None
            driver.get(url)
            try:
                async_wait_first.until(expected_conditions.presence_of_element_located((By.XPATH, self.config["content-xpath"])))
            except TimeoutException: # first timeout, try again
                try:
                    async_wait_second.until(expected_conditions.presence_of_element_located((By.XPATH, self.config["content-xpath"])))
                except TimeoutException: # second timeout, raise error
                    raise TimeoutException(err_time.format(PAGE_LOAD_TIME, url, self.source))
            else:
                driver.execute_script("window.stop();")
                elem = driver.find_element(By.XPATH, self.config["content-xpath"])

                return elem
        return wrapped_check

    @classmethod
    def spawn(cls, src_name, query_kws):
        """ Class factory that spawns our instances

        src_name: valid news source name compatible with news_api
        """
        return cls(src_name, query_kws)

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
            thread_swap(1)

    def _fetch_website(self, query):
        print(query['url'])

        return self.driver(query['url'])

    def _cleanup_content(self, content_raw):
        # TODO: add lxml tree and xpaths for content removal
        return content_html

    def _process(self, query, content) -> dict:
        query['article'] = content

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
            if query['url'] not in self.seen_sites:
                content_raw = self._fetch_website(query)
                if content_raw is None:
                    print("Content not found for {} on {}".format(query['url'], self.source))
                    continue
                content_raw = content_raw.get_attribute('outerHTML')
                content = self._cleanup_content(content_raw)
                processed_data = self._process(query, content)
                self.save_article(**processed_data)
                thread_swap(0.1)
            else:
                print("Skipping {}".format(query['url']))

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
            q = self.api.get_everything(q=','.join(self.query_kws),
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
    scraper = NewsAPIScraper('crypto-coins-news', ['bitcoin'])  # Test name
