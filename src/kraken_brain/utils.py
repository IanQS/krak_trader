"""
kraken_brain/Utils.py - Utils agnostic to models or tasks

Author: Ian Q.

Notes:

"""

import os

import numpy as np
import tensorflow as tf
from typing import Tuple

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries/{}'.format(name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

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


def toy_data_generator(batch_size: int, n_steps: int, predict_ahead: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Generates toy data from a sequence of integers

    [Batch_size, n_steps, 1] -> Assuming that we have 1 feature

    :param batch_size:
    :param n_steps: number of time steps.
    :param predict_ahead:
    :return:
    """
    start = np.random.randint(low=0, high=100, size=(batch_size, 1))
    sequences = start + np.arange(0, n_steps + predict_ahead)
    X = np.expand_dims(sequences[:, :-predict_ahead], -1)
    y = np.expand_dims(sequences[:, predict_ahead:], -1)
    return X, y