"""
model.py - Central brain
Goal:
    Input: orderbook
    output: vector predicting future % change at each step

Author: Ian Q.

Notes:
    None
"""

from kraken_brain.trader_configs import LARGE_TEST_FILE

class Brain(object):
    """
    Acts as the central brain. Takes market depth chart, and other features
    """
    def __init__(self, data_path: list):
        self.load_path = data_path

        self.train, self.predict = self._construct_model()

    @staticmethod
    def _construct_model(self):
        return None, None


if __name__ == '__main__':
    f_ = LARGE_TEST_FILE
    model = Brain([f_])