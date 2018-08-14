"""
lstm.py -> basic LSTM


Author: Ian Q.

Notes:
    We don't use the base model (kraken_brain.linear_models._models.base_model.BaseModel)
    because the base model is more for predictors we can spawn up and train quickly

"""
import tensorflow as tf

# Get price data
from kraken_brain.trader_configs import ALL_DATA
from kraken_brain.linear_models.utils import get_price_data
from kraken_brain.nonlinear_models.utils import construct_windows

# Rolling window
WiNDOW_X = 15
WINDOW_Y = 3

TEST = False

class LSTM_Regressor(object):
    def __init__(self, sess):
        self.sess = sess
        n_steps = seq_len - 1
        n_inputs = 4
        n_neurons = 200
        n_outputs = 4
        n_layers = 2
        learning_rate = 0.001
        batch_size = 50
        n_epochs = 100

        self._initialize_model()
        self.train_op = self._construct_training()

    def _initialize_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.steps, self.inputs])
        self.y = tf.placeholder(tf.float32, [None, self.outputs])




        self.outputs = None

    def _construct_training(self):
        loss = tf.reduce_mean(tf.square(outputs - self.y))  # loss function = mean squared error
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)
        return training_op


if __name__ == "__main__":
    ################################################
    # Grab training data
    ################################################
    data = get_price_data(ALL_DATA, 'XXRPZUSD')
    processed_data = [datum[:, 0] for datum in data]
    x, y = construct_windows(processed_data, WiNDOW_X, WINDOW_Y, normalize=True)
    print('Training samples: {}'.format(len(x)))


    if TEST:
        test_data = get_price_data(ALL_DATA, 'EOSUSD')
        test_processed_data = [datum[:, 0] for datum in test_data]
        test_x, test_y = construct_windows(test_processed_data, WiNDOW_X, WINDOW_Y, normalize=True)
