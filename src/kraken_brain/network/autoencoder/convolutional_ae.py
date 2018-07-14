"""
Conv auto-encoder, then we remove the decoder and use the
generated

Author: Ian Q
"""
from kraken_brain.network.autoencoder.autoencoder_base import Autoencoder
import tensorflow as tf
from kraken_brain.trader_configs import ALL_DATA, CONV_INPUT_SHAPE
from kraken_brain.utils import get_image_from_np, custom_scale, split_data


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
    def _train_construction(self):
        base_loss = tf.losses.mean_squared_error(labels=self.encoder_input, predictions=self.decoder)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([base_loss] + reg_losses, name="loss")

        cost = tf.reduce_mean(loss)

        tf.summary.scalar('cost', cost)
        optimizer = tf.train.AdamOptimizer(self.lr)

        grads = optimizer.compute_gradients(cost)
        # Update the weights wrt to the gradient
        optimizer = optimizer.apply_gradients(grads)
        # Save the grads with tf.summary.histogram
        for index, grad in enumerate(grads):
            tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])

        validation_score = tf.metrics.mean_squared_error(labels=self.encoder_input, predictions=self.decoder)
        tf.summary.scalar("Validation Error", validation_score[0])  # as tf.metrics.mse returns tuple

        im_plot = (self.decoder - self.encoder_input)[0:1]
        asks = im_plot[:, :, :, 0:1]
        bids = im_plot[:, :, :, 1:2]
        tf.summary.image('asks', asks)
        tf.summary.image('bids', bids)
        return cost, optimizer, validation_score, im_plot


if __name__ == '__main__':
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    graph = tf.Graph()
    BATCH_SIZE = 256
    CURRENCY = 'XXRPZUSD'
    EPOCHS = 50
    model = ConvolutionalAE(sess, graph, CONV_INPUT_SHAPE, BATCH_SIZE, debug=True, epochs=EPOCHS)

    ################################################
    # Data processing steps
    ################################################
    data = get_image_from_np(ALL_DATA, CURRENCY)
    # Scale the data axis-wise
    data = custom_scale(data, (data[0].shape[0], *CONV_INPUT_SHAPE))
    data, validation_data = split_data(data, BATCH_SIZE * 100, maintain_temporal=False)
    # train, then validate
    model.train(data, validation_data)
