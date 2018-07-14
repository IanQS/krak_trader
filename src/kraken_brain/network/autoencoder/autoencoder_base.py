"""
Base autoencoder class. Implements most things generically, so the user should
only need to override

Author: Ian Q
"""
import tensorflow as tf
import numpy as np
from kraken_brain.trader_configs import SUMMARY_PATH
from kraken_brain.utils import clear_tensorboard
from abc import ABC, abstractmethod
from tqdm import trange


class Autoencoder(ABC):
    """
    Fully Connected Autoencoder.

    In theory, all the steps should be the same except for construct encoder and construct decoder
    which you can overwrite
    """
    def __init__(self, sess: tf.InteractiveSession, graph: tf.Graph, input_shape: tuple, batch_size: int,
                 debug: bool, lr: float = 0.1, epochs=20, name=None):
        self.sess = sess
        self.graph = graph
        self.batch_size = batch_size
        self.shape = (batch_size, *input_shape)
        self.lr = lr
        self.epochs = epochs

        ################################################
        # Construct Graph
        ################################################
        with tf.variable_scope('autoencoder'):
            self.encoder_input = None
            self.encoder = self._construct_encoder(self.shape)
            self.decoder = self._construct_decoder(self.encoder)
            self.cost, self.train_op, self.validation = self._train_construction

        ################################################
        # Construct loggers
        ################################################
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        summary_path = SUMMARY_PATH.format(self.__class__.__name__ if name is None else name)

        if debug:  # Clear contents of summary_path if there are contents
            clear_tensorboard(summary_path)
        self.file_writer = tf.summary.FileWriter(summary_path, self.sess.graph)
        self.summ = tf.summary.merge_all()

    @abstractmethod
    def _construct_encoder(self, input_shape: tuple):
        raise NotImplementedError

    @abstractmethod
    def _construct_decoder(self, encoded):
        raise NotImplementedError

    @property
    def _train_construction(self):
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

        validation_score = tf.metrics.mean_squared_error(labels=self.encoder_input, predictions=self.decoder)
        tf.summary.scalar("Validation Error", validation_score[0])  # as tf.metrics.mse returns 2 vals
        return cost, optimizer, validation_score

    def train(self, orderbook_data, validation_data):
        total_runs = 0
        for i in trange(self.epochs, desc='Epochs'):

            ################################################
            # Train the model
            ################################################
            batch_length = len(orderbook_data)
            randomized = np.random.choice(batch_length, batch_length, replace=False)
            for mb in trange(batch_length // self.batch_size, desc='Minibatch', leave=True):
                total_runs += 1
                minibatch = orderbook_data[randomized[mb * self.batch_size: (mb + 1) * self.batch_size], :]
                summary, x = self.sess.run([self.summ, self.train_op],
                                           feed_dict={self.encoder_input: minibatch})
                self.file_writer.add_summary(summary, total_runs)

            ################################################
            # Validate on fixed number of points
            ################################################
            validation = validation_data[np.random.choice(len(validation_data), self.batch_size)]
            score = self.sess.run(self.validation, feed_dict={self.encoder_input: validation})
            print("\nValidation Error: Step {} Error: {}\n".format(i, score))
