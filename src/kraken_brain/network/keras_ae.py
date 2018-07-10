"""
Autoencoder sanity check with keras

Author: Ian Q
"""
import keras
from kraken_brain.trader_configs import ALL_DATA
from kraken_brain.utils import get_image_from_np

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K

if __name__ == '__main__':
    BATCH_SIZE = 100
    CURRENCY = 'XXRPZUSD'

    data = [get_image_from_np(ALL_DATA, CURRENCY)]

    input_img = Input(shape=(100, 4))
    flattened = Flatten()(input_img)
    encoded = Dense(128, activation='relu')(flattened)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    encoded = BatchNormalization()(encoded)
    decoded = Dense(400, activation=None)(decoded)
    reshaped = Reshape((100, 4))(decoded)

    autoencoder = Model(input_img, reshaped)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(data, data,
                    epochs=50,
                    batch_size=256,
                    shuffle=True, )