import sys

import numpy as np
import pandas as pd

def get_price_data(data_path: str, currency: str):
    """ Takes in data in a list format, and currency in str, then
    unpacks the np data into a format we can use

    Returns
        [p_0, p_1, v_0, v_1] where:
        p = <today>, <last 24 hours>
        v = <today>, <last 24 hours>

    """
    all_data = [np.load(f)[currency] for f in data_path if f.endswith('npz')]
    price_volume = []
    for datum in all_data:
        for ind in datum:
            price_volume.append(
                np.asarray([ind['prices'], ind['volumes']])
            )
    return np.asarray(price_volume)

def process_data(data):
    pass