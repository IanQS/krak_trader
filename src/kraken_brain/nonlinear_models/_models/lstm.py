"""
lstm.py -> basic LSTM

Largely inspired by
    Stanford CS20:
        RNN lecture
        https://web.stanford.edu/class/cs20si/syllabus.html

Author: Ian Q.

Notes:
    We don't use the base model (kraken_brain.linear_models._models.base_model.BaseModel)
    because the base model is more for predictors we can spawn up and train quickly

"""
import tensorflow as tf

# Get price data
from kraken_brain.trader_configs import ALL_DATA, LSTM_CONFIG
from kraken_brain.linear_models.utils import get_price_data
from kraken_brain.nonlinear_models.utils import construct_windows

# Rolling window
WINDOW_X = 15
WINDOW_Y = 3

TEST = False

class LSTM_Regressor(object):
    def __init__(self, sess, config):
        self.sess = sess
        config.overwrite(dict(x_window_size=WiNDOW_X, y_window_size=WINDOW_Y))
        for k, v in config.expose():
            setattr(self, k, v)
        self.config = config

        self._construct_training_ops()


    def _construct_training_ops(self):
        self.x = tf.placeholder(tf.float32, [None, self.x_window_size])
        self.y = tf.placeholder(tf.float32, [None, self.y_window_size])
        self.lr = tf.placeholder(tf.float32, None)

    def _construct_model(self):
        hidden_units = self.hidden_units
        if isinstance(hidden_units, int):
            hidden_units = [self.hidden_units for _ in range(len(self.num_layers))]

        if self.num_layers == 1:
            cells = tf.nn.rnn_cell.LSTMCell(hidden_units[0], activation=tf.nn.relu)
        else:
            layers = [tf.nn.rnn_cell.GRUCell(size) for size in hidden_units]
            cells = tf.nn.rnn_cell.MultiRNNCell(layers)
        output, output_states = tf.nn.dynamic_rnn(cells, self.x, dtype=tf.float32)

    def _construct_training(self):


if __name__ == "__main__":
    ################################################
    # Grab training data
    ################################################
    data = get_price_data(ALL_DATA, 'XXRPZUSD')
    processed_data = [datum[:, 0] for datum in data]
    x, y = construct_windows(processed_data, WINDOW_X, WINDOW_Y, normalize=True)
    print('Training samples: {}'.format(len(x)))

    ################################################
    # Training procedure
    ################################################

    config = LSTM_CONFIG()



    if TEST:
        test_data = get_price_data(ALL_DATA, 'EOSUSD')
        test_processed_data = [datum[:, 0] for datum in test_data]
        test_x, test_y = construct_windows(test_processed_data, WINDOW_X, WINDOW_Y, normalize=True)
