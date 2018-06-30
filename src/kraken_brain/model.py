"""
model.py - Central brain
Goal:
    Input: orderbook
    output: vector predicting future % change at each step

Author: Ian Q.

Notes:
    None
"""

import tensorflow as tf

from kraken_brain.trader_configs import TEST_FILE
from kraken_brain.network.convolutional import Convolutional

class Brain(object):
    """
    Acts as the central brain. Takes market depth chart, and other features
    """
    def __init__(self, data_path: list):
        self.load_path = data_path

        self.train, self.predict = self._construct_model()

    def _construct_model(self):

        return None, None


if __name__ == '__main__':
    f_ = TEST_FILE
    model = Brain([f_])