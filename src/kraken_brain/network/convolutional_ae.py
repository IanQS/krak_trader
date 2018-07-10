"""
Conv auto-encoder, then we remove the decoder and use the
generated

Author: Ian Q
"""
from kraken_brain.network.autoencoder_base import Autoencoder
import tensorflow as tf
from kraken_brain.trader_configs import ALL_DATA
from kraken_brain.utils import get_image_from_np, custom_scale


class ConvolutionalAE(Autoencoder):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = self.__class__.__name__
        super().__init__(*args, **kwargs)

    def _construct_encoder(self, input_shape: tuple):
        """
        :param input_shape: # 100 x 2 x 2 (OB, price-vol, bids-asks)
        :return:
        """
        self.encoder_input = tf.placeholder(tf.float32, input_shape, name='x')
        with tf.variable_scope("encoder"):
            flattened = tf.layers.flatten(self.encoder_input)
            e_fc_1 = tf.layers.dense(flattened, units=150, activation=tf.nn.relu)
            encoded = tf.layers.dense(e_fc_1, units=75, activation=None)
            # Now 25x1x16
            return encoded

    def _construct_decoder(self, encoded):
        with tf.variable_scope("decoder"):
            d_fc_1 = tf.layers.dense(encoded, 150, activation=tf.nn.relu)
            d_fc_2 = tf.layers.dense(d_fc_1, 400, activation=None)
            decoded = tf.reshape(d_fc_2, self.shape)
            return decoded


if __name__ == '__main__':
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    graph = tf.Graph()
    BATCH_SIZE = 256
    CURRENCY = 'XXRPZUSD'
    EPOCHS = 50
    model = ConvolutionalAE(sess, graph, (100, 4), BATCH_SIZE, debug=True, epochs=EPOCHS)

    ################################################
    # Data processing steps
    ################################################
    data = get_image_from_np(ALL_DATA, CURRENCY)
    # Scale the data axis-wise
    data = custom_scale(data, (data[0].shape[0], 100, 2 * 2))
    validation_data, data = data[:BATCH_SIZE, :],  data[BATCH_SIZE:, :]

    # train, then validate
    model.train(data, validation_data)
