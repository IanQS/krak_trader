"""
Trained as a convolutional auto-encoder, then we remove the decoder and use the
generated

Author: Ian Q
"""
import sys

import tensorflow as tf
import numpy as np
from kraken_brain.trader_configs import ALL_DATA
from kraken_brain.utils import get_image_from_np

class ConvAE(object):
    def __init__(self, sess: tf.Session, graph: tf.Graph, input_shape: tuple, batch_size: int,
                 lr: float = 0.1):
        self.sess = sess
        self.graph = graph
        self.shape = [batch_size, *input_shape]
        self.lr = lr

        ################################################
        # Construct Graph
        ################################################
        self.encoder = self.__construct_encoder(self.shape)
        self.decoder = self.__construct_decoder(self.encoder)
        self.cost, self.train_op = self.train()

    def __construct_encoder(self, input_shape: tuple):
        """
        :param input_shape: # 100 x 2 x 2 (OB, price-vol, bids-asks)
        :return:
        """
        self.encoder_input = tf.placeholder(tf.float32, input_shape, name='x')
        with self.graph.name_scope("encoder"):
            conv_e_1 = tf.layers.conv2d(inputs=self.encoder_input, filters=64, kernel_size=(2, 2), padding='same',
                                        activation=tf.nn.relu)
            # (batch_size, 100, 2, 64)
            maxpool1 = tf.layers.max_pooling2d(conv_e_1, pool_size=(2, 2), strides=(2, 2), padding='same')
            # (batch_size, 50, 1, 64)
            conv_e_2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(2, 2), padding='same',
                                      activation=tf.nn.relu)
            # (batch_size, 50, 1, 32)
            maxpool2 = tf.layers.max_pooling2d(conv_e_2, pool_size=(2, 1), strides=(2, 2), padding='same')
            # (batch_size, 25, 1, 32)
            return maxpool2

    def __construct_decoder(self, last_encoder_op):
        """
        :param input_shape: # 100 x 2 x 2 (OB, price-vol, bids-asks)
        :return:
        """
        with self.graph.name_scope("decoder"):
            upsample1 = tf.image.resize_images(last_encoder_op, size=(50, 1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # (batch_size, 50, 1, 32)
            conv_d_1 = tf.layers.conv2d(inputs=upsample1, filters=64, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            # (batch_size, 50, 1, 64)
            upsample2 = tf.image.resize_images(conv_d_1, size=(100, 2),
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # (batch_size, 100, 2, 64)
            conv_d_2 = tf.layers.conv2d(inputs=upsample2, filters=2, kernel_size=(3, 3), padding='same',
                                        activation=tf.nn.relu)
            # (batch_size, 100, 2, 2)
            return conv_d_2

    def train(self):
        self.y_prime = tf.placeholder(tf.float32, self.shape, name='y_prime')
        pred = tf.nn.sigmoid(self.decoder)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_prime, logits=pred)

        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(self.lr).minimize(cost)
        return cost, optimizer

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    graph = tf.Graph()
    data = np.random.rand(100, 2, 2)
    BATCH_SIZE = 100
    CURRENCY = 'XXRPZUSD'
    model = ConvAE(sess, graph, data.shape, BATCH_SIZE)
    sess.run(tf.global_variables_initializer())

    orderbook_data = [get_image_from_np(ALL_DATA, CURRENCY )]

    for i in range(len(orderbook_data) * 50):
        curr_data = orderbook_data[i % len(orderbook_data)]
        data_shape = curr_data.shape[0]
        randomized = np.random.choice(data_shape, data_shape, replace=False)
        for _ in range(data_shape // BATCH_SIZE):
            minibatch = curr_data[randomized[_ * BATCH_SIZE: (_ + 1) * BATCH_SIZE ], :]
            cost, x = sess.run([model.cost, model.train_op], feed_dict={model.encoder_input: minibatch, model.y_prime: minibatch})
            print('{}: {}'.format(cost, x))
