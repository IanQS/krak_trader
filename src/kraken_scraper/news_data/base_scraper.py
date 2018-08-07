"""
base_scraper
  - all scrapers should inherit from this and utilize the provided functions

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
from constants import NEWS_PATH

import numpy as np


class GenericScraper(ABC):
    def __init__(self, site_dir: str, ):
        """ Inherited init method
        :param site_dir: directory to save data to
        :param seen_urls: empty dict (new scraper), OR set data loaded in
        """
        self.storage_absolute = NEWS_PATH.format(site_dir)
        self.name = site_dir
        self.seen_sites, self._seen_sites_store = self._storage_site_setup()
        self.parser = regexp.compile('(\d{2}\-[a-zA-Z]*\-\d{4}|\d{4}\-\d{2}\-\d{2})')

    def _storage_site_setup(self):
        """ Construct the storage site (if not exists), and initiate
        the 'seen sites' file from which we store URLs
        """
        seen_file = '/seen_data.txt'
        seen_sites_store = self.storage_absolute + seen_file
        if not os.path.isdir(self.storage_absolute):
            os.mkdir(self.storage_absolute)

            # Create the storage file too
            with open(seen_sites_store, 'a') as f:
                f.close()
        with open(seen_sites_store, 'r') as f:
            data = f.readlines()
            data = [datum.strip() for datum in data]
            return set(data), seen_sites_store

    def save_article(self, article, url:str, date:str, site:str, author: str = None):
        f_name = url.replace("/", "=")
        with open(self._seen_sites_store, 'a') as f:
            f.write(f_name)

        if not self.parser.match(date):
            raise ValueError('Date not formatted correctly')

        f_path = os.path.join(self.storage_absolute, f_name)
        data = {"article": article,
                "url": url,
                "date": date,
                "site": site,
                "author": author}
        np.savez(f_path, **data)