"""
base_scraper

Useful links:
    - https://realpython.com/python-web-scraping-practical-introduction/
    - https://automatetheboringstuff.com/chapter11/


Author: Ian Q.

Notes:
    Inherit the public method which is a type-checker and generic base,
    and make sure you overwrite the semi-private method
"""
import os
from abc import ABC, abstractmethod
import re as regexp

import numpy as np


class GenericScraper(ABC):
    def __init__(self, website: str, site_dir: str, seen_urls: set = None):
        """ Inherited method

        :param website: website to scrape from
        :param site_dir: directory to save data to
        :param seen_urls: empty dict (new scraper), OR set data loaded in
        """
        self.url = website
        self.storage_site = site_dir
        self.seen_sites = seen_urls if seen_urls else {}
        self.parser = regexp.compile('\d{2}\-[a-zA-Z]*\-\d{4}')

    def parse_url(self, url):
        pass

    @abstractmethod
    def _parse_url(self):
        raise NotImplementedError

    def save_article(self, article, url, date):
        with open(os.path.join(self.storage_site, 'seen_sites.txt'), 'a') as f:
            f.write(url)

        if not self.parser.match(date):
            raise ValueError('Data not formatted correctly')

        f_name = os.path.join(self.storage_site, url)
        np.savez(f_name, article)

    @abstractmethod
    def _save_article(self):
        raise NotImplementedError
