"""
Trained as a convolutional auto-encoder, then we remove the decoder and use the
generated

Author: Ian Q
"""
import sys
import time

import tensorflow as tf
import numpy as np
from kraken_brain.trader_configs import ALL_DATA, SUMMARY_PATH, CONV_INPUT_SHAPE
from kraken_brain.utils import get_image_from_np, clear_tensorboard

class ConvAE(object):
    def __init__(self, sess: tf.Session, graph: tf.Graph, input_shape: tuple, batch_size: int,
                 debug: bool, lr: float = 0.8):
        self.sess = sess
        self.graph = graph
        self.batch_size = batch_size
        self.shape = [batch_size, *input_shape]
        self.lr = lr


        ################################################
        # Construct Graph
        ################################################
        with tf.variable_scope('convolutional_autoencoder'):
            self.encoder = self.__construct_encoder(self.shape)
            self.decoder = self.__construct_decoder(self.encoder)
            self.cost, self.train_op = self.__train_op

        ################################################
        # Construct loggers
        ################################################
        self.sess.run(tf.global_variables_initializer())
        summary_path = SUMMARY_PATH.format(self.__class__.__name__)
        if debug:
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
            c1 = conv2d(input, name='c1', kshape=[5, 5, 1, 25])
            p1 = maxpool2d(c1, name='p1')
            do1 = dropout(p1, name='do1', keep_rate=0.75)
            do1 = tf.reshape(do1, shape=[-1, 14 * 14 * 25])
            fc1 = fullyConnected(do1, name='fc1', output_size=14 * 14 * 5)
            do2 = dropout(fc1, name='do2', keep_rate=0.75)
            fc2 = fullyConnected(do2, name='fc2', output_size=14 * 14)

    def __construct_decoder(self, last_encoder_op):
        """
        :param input_shape: # 100 x 2 x 2 (OB, price-vol, bids-asks)
        :return:
        """
        with tf.variable_scope("decoder"):
            fc3 = fullyConnected(fc2, name='fc3', output_size=14 * 14 * 5)
            do3 = dropout(fc3, name='do3', keep_rate=0.75)
            fc4 = fullyConnected(do3, name='fc4', output_size=14 * 14 * 25)
            do4 = dropout(fc4, name='do3', keep_rate=0.75)
            do4 = tf.reshape(do4, shape=[-1, 14, 14, 25])
            dc1 = deconv2d(do4, name='dc1', kshape=[5, 5], n_outputs=25)
            up1 = upsample(dc1, name='up1', factor=[2, 2])

            output = fullyConnected(up1, name='output', output_size=28 * 28)

    @property
    def __train_op(self):
        self.y_prime = tf.placeholder(tf.float32, self.shape, name='y_prime')
        reconstruction = self.decoder
        loss = tf.nn.l2_loss(self.y_prime - reconstruction)
        cost = tf.reduce_mean(loss)

        tf.summary.scalar('cost', cost)
        optimizer = tf.train.AdamOptimizer(self.lr).minimize(cost)
        return cost, optimizer

    def train(self, orderbook_data):
        total_runs = 0
        for i in range(len(orderbook_data) * 2):
            curr_data = orderbook_data[i % len(orderbook_data)]
            data_shape = curr_data.shape[0]
            randomized = np.random.choice(data_shape, data_shape, replace=False)
            for j, mb in enumerate(range(data_shape // BATCH_SIZE)):
                total_runs += 1
                minibatch = curr_data[randomized[mb * BATCH_SIZE: (mb + 1) * BATCH_SIZE], :]
                summary, x = sess.run([self.summ, model.train_op],
                                   feed_dict={model.encoder_input: minibatch, model.y_prime: minibatch})
                self.file_writer.add_summary(summary, total_runs)




if __name__ == '__main__':
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    graph = tf.Graph()
    BATCH_SIZE = 100
    CURRENCY = 'XXRPZUSD'
    model = ConvAE(sess, graph, CONV_INPUT_SHAPE, BATCH_SIZE, debug=True)


    orderbook_data = [get_image_from_np(ALL_DATA, CURRENCY )]
    model.train(orderbook_data)