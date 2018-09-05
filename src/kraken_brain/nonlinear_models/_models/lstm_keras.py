import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Get price data
from kraken_brain.trader_configs import ALL_DATA, LSTM_CONFIG
from kraken_brain.linear_models.utils import get_price_data
from kraken_brain.nonlinear_models.utils import construct_windows

# Rolling window
WINDOW_X = 30
WINDOW_Y = 3
TEST = True

class LSTM(object):
    def __init__(self, batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self._construct_model()


    def _construct_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(None, WINDOW_X)))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model


    def train(self, train_X, train_Y):
        self.model.fit(train_X, train_Y, epochs=self.epochs, batch_size=self.batch_size, verbose=2)

    def predict(self, test_X):
        self.model.predict(test_X)


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


    if TEST:
        test_data = get_price_data(ALL_DATA, 'EOSUSD')
        test_processed_data = [datum[:, 0] for datum in test_data]
        test_x, test_y = construct_windows(test_processed_data, WINDOW_X, WINDOW_Y, normalize=True)
