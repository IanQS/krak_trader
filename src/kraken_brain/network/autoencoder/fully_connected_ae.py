"""
Vanilla auto-encoder, then we remove the decoder and use the
generated

Author: Ian Q
"""
from kraken_brain.network.autoencoder.autoencoder_base import Autoencoder
import tensorflow as tf
from kraken_brain.trader_configs import ALL_DATA
from kraken_brain.utils import get_image_from_np, custom_scale, split_data, variable_summaries, ob_diff


class FullyConnectedAE(Autoencoder):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = self.__class__.__name__ + '_diff'
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

    @property
    def train_construction(self):
        loss = tf.losses.mean_squared_error(labels=self.encoder_input, predictions=self.decoder)
        cost = tf.reduce_mean(loss)
        return super()._train_construction(cost)

    def contextual_magnitude(self, val):
        asks_p = val[:, :, 0:1]
        asks_v = val[:, :, 1:2]
        bids_p = val[:, :, 3:4]
        bids_v = val[:, :, 4:5]
        variable_summaries(asks_p, 'ask_price')
        variable_summaries(asks_v, 'ask_vol')
        variable_summaries(bids_p, 'bid_price')
        variable_summaries(bids_v, 'bid_vol')

if __name__ == '__main__':
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    graph = tf.Graph()
    BATCH_SIZE = 256
    CURRENCY = 'XXRPZUSD'
    EPOCHS = 15
    model = FullyConnectedAE(sess, graph, (100, 4), BATCH_SIZE, debug=True, epochs=EPOCHS)

    ################################################
    # Data processing steps
    ################################################
    data = get_image_from_np(ALL_DATA, CURRENCY)
    # Scale the data axis-wise
    data = custom_scale(data, (-1, 100, 2 * 2))
    data, validation_data = split_data(data, BATCH_SIZE * 100, maintain_temporal=False)
    # train, then validate
    model.train(data, validation_data)
