import sys
import time
import numpy as np

import krakenex
from kraken_scraper.scraper_constants import STORAGE_SIZE, PAIRS_TO_STORE
from constants import STORAGE_PATH
from tqdm import tqdm


class Scraper(object):
    def __init__(self, pairs: list, path: str):
        self.pairs = pairs
        self.path = path
        self.scraper = krakenex.API(key='', secret='')
        self.start = None

    def scrape(self):
        processed_data = {k: [] for k in self.pairs}
        processed_data['Time'] = []
        insertions = 0
        self.start = time.time()
        display = tqdm(total=STORAGE_SIZE)
        while True:
            if insertions % (0.05 * STORAGE_SIZE) == 0 and insertions != 0:  # Update TQDM every 5 %
                display.update(insertions)
            res = self._grab_data()
            if res and not res['error']:
                if insertions > STORAGE_SIZE:
                    processed_data, insertions = self.write_data(processed_data)
                    print('Time taken: {}s'.format(time.time() - self.start))
                    self.start = time.time()
                    insertions = 0
                else:
                    _process = self.process_data(res)
                    for k in _process.keys():
                        processed_data[k] += [_process[k]]
                    insertions += 1
            else:
                self._cleanup(res)
                display.close()

    def write_data(self, processed_data: dict) -> tuple:
        processed_data = {k: np.asarray(v) for k, v in processed_data.items()}
        np.savez_compressed(self.path.format(time.time()), **processed_data)
        processed_data = {k: [] for k in processed_data.keys()}
        return processed_data, 0

    def _cleanup(self, res: dict) -> None:
        err_msg = 'Scraped data error: res: {}, res[error]: {}'
        errors = (res, res['error'] if res else None)
        print(err_msg.format(*errors))
        self.scraper.close()
        sys.exit(0)

    def _grab_data(self) -> dict:
        res = None
        try:
            res = self.scraper.query_public('Ticker', {'pair': ','.join(self.pairs)})
        except Exception as e:
            print('Error querying: {}'.format(e))
            time.sleep(1)
            res = self.scraper.query_public('Ticker', {'pair': ','.join(self.pairs)})
        return res

    def process_data(self, data_res: dict):
        """
        v = volume array(<today>, <last 24 hours>),
        p = volume weighted average price array(<today>, <last 24 hours>),

        :param data_res:
        :return:
            dict
        """
        ret = {}
        for k, v in data_res['result'].items():
            ret[k] = [*[float(prices) for prices in v['p']], * [float(volumes) for volumes in v['v']]]  #
        ret['Time'] = time.time()
        return ret


if __name__ == '__main__':
    scraper = Scraper(PAIRS_TO_STORE, STORAGE_PATH)
    scraper.scrape()
