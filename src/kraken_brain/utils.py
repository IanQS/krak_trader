import numpy as np
import os
from sklearn.preprocessing import minmax_scale, robust_scale, normalize


def get_image_from_np(data_path: str, currency: str) -> list:
    """ Takes in data in a list format, and currency in str, then
    unpacks the np data into a format we can use

    :param data:
    :param currency:
    :return:
    """
    all_data = [np.load(f)[currency] for f in data_path if f.endswith('npz')]
    orderbook = []
    for datum in all_data:
        for ind in datum:
            orderbook.append(
                np.stack([ind['asks'], ind['bids']], axis=-1)
            )
    orderbook = np.asarray(orderbook)
    return [orderbook]

def custom_scale(data: np.array, final_shape: tuple) -> np.ndarray:
    if isinstance(data, list):
        data = data[0]
    holder = []
    for outer in [0, 1]:  # bids or asks
        for inner in [0, 1]:  # price or vol
            data_ = data[:, :, inner, outer]
            holder.append(normalize(data_))
    data = np.moveaxis(np.asarray(holder), 0, -1)
    return data.reshape(final_shape)

def clear_tensorboard(path: str):
    path = path.replace('.', '')
    path = os.path.expanduser(os.getcwd() + path)
    if not os.path.exists(path):
        return
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def split_data(data: np.array, batch_size: int, maintain_temporal: bool = True) -> tuple:
    """ Splits data into training and validation. Currently only supports
    returning validation size of batch_size.

    Supports both RNN and CNN. maintain_temporal assures that whatever index we choose from,
    validation = data[x: x + batch_size] for RNN
    """
    if maintain_temporal:
        indices = np.random.choice(np.arange(len(data) - batch_size), size=1)
        indices = np.arange(indices, indices + batch_size)
    else:
        indices = np.random.choice(np.arange(len(data)), batch_size)
    indices_complement = np.delete(np.arange(len(data)), np.r_[indices])

    print('Train-Val Split: {}-{}'.format(len(indices_complement) // len(indices), 1))
    return data[indices_complement], data[indices]