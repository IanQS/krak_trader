"""
Conv auto-encoder, then we remove the decoder and use the
generated

Author: Ian Q
"""
from .autoencoder_base import Autoencoder
import tensorflow as tf
from kraken_brain.trader_configs import ALL_DATA, CONV_INPUT_SHAPE
from kraken_brain.utils import variable_summaries
from kraken_brain.video_to_predictions.orderbook_embedding.utils import split_data, get_image_from_np, custom_scale


class ConvolutionalAE(Autoencoder):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = self.__class__.__name__
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        super().__init__(*args, **kwargs)

    def _construct_encoder(self, input_shape: tuple):
        self.encoder_input = tf.placeholder(tf.float32, input_shape, name='x')

        with tf.variable_scope("encoder"):
            conv1 = tf.layers.conv2d(self.encoder_input, filters=32, kernel_size=(2, 2),
                                     activation=tf.nn.relu, padding='same', kernel_regularizer=self.regularizer)
            # (#, 100, 2, 32)
            mp1 = tf.layers.max_pooling2d(conv1, pool_size=(4, 1), strides=(1, 1))
            # (#, 25, 2, 32)
            conv2 = tf.layers.conv2d(mp1, filters=64, kernel_size=(2, 2),
                                     activation=None, padding='same', kernel_regularizer=self.regularizer)
            # (#, 25, 2, 64)
            return conv2

    def _construct_decoder(self, encoded):
        with tf.variable_scope("decoder"):
            upsample1 = tf.image.resize_images(encoded, size=(50, 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # (#, 50, 2, 64)
            conv4 = tf.layers.conv2d(inputs=upsample1, filters=32, kernel_size=(2, 2), padding='same',
                                     activation=tf.nn.relu, kernel_regularizer=self.regularizer)
            # (#, 50, 2, 32)
            upsample2 = tf.image.resize_images(conv4, size=(100, 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # (#, 100, 2, 32)
            conv5 = tf.layers.conv2d(inputs=upsample2, filters=2, kernel_size=(2, 2), padding='same',
                                     activation=None, kernel_regularizer=self.regularizer)
            # (#, 100, 2, 2)
            return conv5

    @property
    def train_construction(self):
        base_loss = tf.losses.mean_squared_error(labels=self.encoder_input, predictions=self.decoder)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([base_loss] + reg_losses, name="loss")

        cost = tf.reduce_mean(loss)
        return super()._train_construction(cost)

    def contextual_magnitude(self, val):
        asks_p = val[:, :, 0:1, 0:1]
        asks_v = val[:, :, 1:2, 0:1]
        bids_p = val[:, :, 0:1, 1:2]
        bids_v = val[:, :, 1:2, 1:2]
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
    model = ConvolutionalAE(sess, graph, CONV_INPUT_SHAPE, BATCH_SIZE, debug=True, epochs=EPOCHS)

    ################################################
    # Data processing steps
    ################################################
    data = get_image_from_np(ALL_DATA, CURRENCY)
    # Scale the data axis-wise
    data = custom_scale(data, (-1, *CONV_INPUT_SHAPE))
    data, validation_data = split_data(data, BATCH_SIZE * 100, maintain_temporal=False)
    # train, then validate
    model.train(data, validation_data)
