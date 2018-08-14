import sys

import numpy as np
import pandas as pd

def get_price_data(data_path: str, currency: str) -> list:
    """ Takes in data in a list format, and currency in str, then
    unpacks the np data into a format we can use

    We assume that the npz files are "independent" of one another - no contiguity is assumed, so all chunks
    are constructed within a sample

    Returns
        [p_0, p_1, v_0, v_1] where:
        p = <today>, <last 24 hours>
        v = <today>, <last 24 hours>

    """
    all_data = [np.load(f)[currency] for f in data_path if f.endswith('npz')]
    price_volume = []
    for datum in all_data:
        subrun = []
        for i, ind in enumerate(datum):
            subrun.append(
                np.asarray([*ind['prices'], *ind['volumes']])
            )
        price_volume.append(np.asarray(subrun))
    return price_volume

def process_data(data):
    fake_example = data[0, :], data[1, :], data[2, :], data[3, :]
    raise NotImplementedError