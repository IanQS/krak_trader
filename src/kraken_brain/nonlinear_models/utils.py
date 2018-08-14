import numpy as np
from sklearn.preprocessing import normalize


def ob_diff(data, final_shape=None, percentagize=False):
    assert final_shape is not None
    if isinstance(data, list):
        data = data[0]
    diff = np.diff(data, axis=0)
    if percentagize:
        return (diff / data[:-1]).reshape(final_shape)
    return diff.reshape(final_shape)


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

def construct_windows(data: np.array, x_window_len: int, y_window_len: int, diff: bool, normalize = False):
    """
    Construct frames of window_length from data. We normalize all the values according
    to the first value

    """
    if not isinstance(data, np.array):
        data = np.asarray(data)
    assert len(data.shape) == 1, 'construct_window only supports vectors for now'

    X_data = []
    y_data = []
    for i in range(len(data) - (x_window_len + y_window_len)):
        x_start, x_end = i, i + x_window_len
        y_start, y_end = x_end, x_end + y_window_len

        x = data[x_start: x_end]
        y = data[y_start: y_end]

        base = x[0]
        if normalize:
            X_data.append((x - base) / base)
            y_data.append((y - base) / base)
        elif diff:
            X_data.append((x - base))
            y_data.append((y - base))
        else:
            X_data.append(x)
            y_data.append(y)

    return np.asarray(X_data), np.asarray(y_data)