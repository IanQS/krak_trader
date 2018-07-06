import sys
import time
import numpy as np

import krakenex
from kraken_scraper.scraper_constants import STORAGE_SIZE, PAIRS_TO_STORE
from constants import STORAGE_PATH
from tqdm import tqdm
import numpy as np


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
            # TQDM update
            if insertions % (0.05 * STORAGE_SIZE) == 0 and insertions != 0:  # Update TQDM every 5 %
                display.update(insertions)

            # Scraping
            res = self._grab_data()
            if res and not res['error']:
                processed_data, insertions = self.process_data(res, processed_data, insertions)
                if insertions > STORAGE_SIZE:
                    processed_data, insertions = self.write_data(processed_data)
            else:
                self._cleanup(res)
                display.close()

    def write_data(self, processed_data: dict) -> tuple:
        processed_data = {k: np.asarray(v) for k, v in processed_data.items()}
        np.savez_compressed(self.path.format(time.time()), **processed_data)
        processed_data = {k: [] for k in processed_data.keys()}

        print('Time taken: {}s'.format(time.time() - self.start))
        self.start = time.time()
        return processed_data, 0

    def _cleanup(self, res: dict) -> None:
        err_msg = 'Scraped data error: res: {}, res[error]: {}'
        errors = (res, res['error'] if res else None)
        print(err_msg.format(*errors))
        self.scraper.close()
        sys.exit(0)

    def _grab_data(self) -> dict:
        res = None

        ################################################
        #Get ticker data
        ################################################
        try:
            res = self.scraper.query_public('Ticker', {'pair': ','.join(self.pairs)})
        except Exception as e:
            print('Error querying Ticker data: {}'.format(e))
            time.sleep(1)
            res = self.scraper.query_public('Ticker', {'pair': ','.join(self.pairs)})
        assert res is not None

        ################################################
        # Get orderbook data
        ################################################
        for k, v in res['result'].items():
            try:
                response = self.scraper.query_public('Depth', {'pair': k, 'count': 100})
            except Exception as e:
                print('Error querying OB: {}'.format(e))
                return {'error': True}  # if we error out on OB query quit since OB is our bread and butter
            res['error'].extend(response['error'])
            result = response['result'][k]
            res['result'][k]['asks'] = np.asarray(result['asks'], dtype=np.float32)[:, :2]
            res['result'][k]['bids'] = np.asarray(result['bids'], dtype=np.float32)[:, :2]
        return res

    def process_data(self, data_res: dict, processed_data: list, insertions: int):
        """
        v = volume array(<today>, <last 24 hours>),
        p = volume weighted average price array(<today>, <last 24 hours>),

        :param data_res:
        :return:
            dict
        """
        ret = {}
        for k, v in data_res['result'].items():
            ret[k] = {}
            ret[k]['prices'] = [float(prices) for prices in v['p']]
            ret[k]['volumes'] = [float(volumes) for volumes in v['v']]
            ret[k]['asks'] = v['asks']
            ret[k]['bids'] = v['bids']
        ret['Time'] = time.time()

        for k in ret.keys():
            processed_data[k] += [ret[k]]
        return processed_data, insertions + 1


if __name__ == '__main__':
    scraper = Scraper(PAIRS_TO_STORE, STORAGE_PATH)
    scraper.scrape()
