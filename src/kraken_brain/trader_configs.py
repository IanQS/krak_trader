from constants import KRAKEN_PATH
import os

ZERO_PLACEHOLDER = 1e-10  # When we try to normalize, we may encounter 0s, so we

SUMMARY_PATH = './summaries/{}'

CONV_INPUT_SHAPE = [100, 2, 2]

small_test_file = ''
SMALL_TEST_FILE = KRAKEN_PATH.format(small_test_file)

large_test_file = '1530759203.3338196.npz'
LARGE_TEST_FILE = KRAKEN_PATH.format(large_test_file)

ALL_DATA = [KRAKEN_PATH.format(f)
            for f in os.listdir(KRAKEN_PATH.split('{}')[0])
            if f.endswith('npz')]

class LSTM_CONFIG(object):
    def __init__(self):
        """

        hidden_units: number of hidden units per layer. If isinstance(int), we propagate shape
        num_layers: number of stacked LSTM layers.
        dropout_keep_prob: drop if less and keep if greater
        lr: learning rate
        learning_rate_decay: decay ratio in later training epochs.
        max_epoch: total number of epochs in training
        window_size: size of the sliding window / one training data point
        batch_size: number of data points to use in one mini-batch.

        """
        config = dict(
            hidden_units=128,
            num_layers=2,
            dropout_keep_prob=0.8,
            lr = 0.001,
            learning_rate_decay=0.99,
            max_epoch=50,
            x_window_size = 30,
            y_window_size=10,
            batch_size=64,
        )

        for k, v in config.items():
            setattr(self, k, v)

        self.config = config

    def overwrite(self, **kwargs):
        for k, v in kwargs.items():
            assert not hasattr(self, k), 'Trying to overwrite non-existent value. Set in LSTM_CONFIG'
            setattr(self, k, v)
            self.config[k] = v

    def expose(self):
        return self.config