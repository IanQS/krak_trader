"""
Trained as a convolutional auto-encoder, then we remove the decoder and use the
generated

Author: Ian Q
"""
import tensorflow as tf
import numpy as np
from kraken_brain.trader_configs import ALL_DATA, SUMMARY_PATH, CONV_INPUT_SHAPE
from kraken_brain.utils import get_image_from_np, clear_tensorboard
from tqdm import tqdm


class Autoencoder(object):
    def __init__(self, sess: tf.InteractiveSession, graph: tf.Graph, input_shape: tuple, batch_size: int,
                 debug: bool, lr: float = 0.1):
        self.sess = sess
        self.graph = graph
        self.batch_size = batch_size
        self.shape = (batch_size, *input_shape)
        self.lr = lr

        ################################################
        # Construct Graph
        ################################################
        with tf.variable_scope('autoencoder'):
            self.encoder = self.__construct_encoder(self.shape)
            self.decoder = self.__construct_decoder(self.encoder)
            self.cost, self.train_op = self.__train_op

        ################################################
        # Construct loggers
        ################################################
        self.sess.run(tf.global_variables_initializer())
        summary_path = SUMMARY_PATH.format(self.__class__.__name__)

        if debug:  # Clear contents of summary_path if there are contents
            clear_tensorboard(summary_path)
        self.file_writer = tf.summary.FileWriter(summary_path, self.sess.graph)
        self.summ = tf.summary.merge_all()

    def __construct_encoder(self, input_shape: tuple):
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

    def __construct_decoder(self, encoded):
        with tf.variable_scope("decoder"):
            d_fc_1 = tf.layers.dense(encoded, 150, activation=tf.nn.relu)
            d_fc_2 = tf.layers.dense(d_fc_1, 400, activation=None)
            decoded = tf.reshape(d_fc_2, self.shape)
            return decoded

    @property
    def __train_op(self):
        loss = tf.losses.mean_squared_error(labels=self.encoder_input, predictions=self.decoder)
        cost = tf.reduce_mean(loss)

        tf.summary.scalar('cost', cost)
        optimizer = tf.train.AdamOptimizer(self.lr)

        grads = optimizer.compute_gradients(cost)
        # Update the weights wrt to the gradient
        optimizer = optimizer.apply_gradients(grads)
        # Save the grads with tf.summary.histogram
        for index, grad in enumerate(grads):
            tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])
        return cost, optimizer

    def train(self, orderbook_data):
        total_runs = 0
        for i in tqdm(range(len(orderbook_data) * 10000)):
            curr_data = orderbook_data[i % len(orderbook_data)]
            data_shape = curr_data.shape[0]
            randomized = np.random.choice(data_shape, data_shape, replace=False)
            for j, mb in enumerate(range(data_shape // BATCH_SIZE)):
                total_runs += 1
                minibatch = curr_data[randomized[mb * BATCH_SIZE: (mb + 1) * BATCH_SIZE], :]
                summary, x = sess.run([self.summ, self.train_op],
                                   feed_dict={self.encoder_input: minibatch})
                self.file_writer.add_summary(summary, total_runs)


if __name__ == '__main__':
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    graph = tf.Graph()
    BATCH_SIZE = 100
    CURRENCY = 'XXRPZUSD'
    model = Autoencoder(sess, graph, CONV_INPUT_SHAPE, BATCH_SIZE, debug=True)

    data = [get_image_from_np(ALL_DATA, CURRENCY)]
    model.train(data)