'''
original author: IanQS
'''

import tensorflow as tf
from abc import ABC, abstractmethod
import numpy as np

# DEBUG = False
# def dprint(s):
#     if DEBUG:
#         print('DEBUG: {}'.format(s))

class Model(ABC):
    """
    Generic base class for models to cleanly structure your project
    """
    def __init__(self, sess, graph, shape, scope=None,learning_rate=0.1, summary_path=".",
                 epochs=20, batch_size=10):
        """ Initializes the model

        :param sess:
        :param graph:
        :param learning_rate:
        :param summary_path:
        """
        self.sess = sess
        self.graph = graph
        self.shape = shape
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        ################################################
        # Construct model Operations
        ################################################
        # dprint("model init began")
        scope = 'Model' if scope is None else scope
        with tf.variable_scope(scope):
            self.encoder_input = None
            self.model = self._construct_model(self.shape)
            self.cost, self.train_op, self.validation_op = self._construct_training(self.model)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        self.file_writer = tf.summary.FileWriter(summary_path, self.sess.graph)
        self.summ = tf.summary.merge_all()
        # dprint("model init finished")

    @abstractmethod
    def _construct_model(self, shape):
        raise NotImplementedError

    @abstractmethod
    def _construct_training(self, model):
        raise NotImplementedError

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def train(self, X, y, test_input, test_output):
        self.total_runs = 0
        for epoch in range(self.epochs):
            randomized = np.random.choice(np.arange(len(X)), len(X), replace=False)
            for mini_b in range((len(X) // self.batch_size) - 1):
                self.total_runs += 1
                print('epoch: {}, mini_b: {}'.format(epoch, mini_b))
                mb_indices = randomized[mini_b * self.batch_size : (mini_b + 1) * self.batch_size]
                assert len(mb_indices) == self.batch_size, 'expected: {}, got: {}'.format(self.batch_size, len(mb_indices))
                
                x_mb = X[mb_indices, :]
                y_mb = y[mb_indices]
                summary, x = self.sess.run([self.summ, self.train_op],
                                           feed_dict={
                                               self.X: x_mb,
                                               self.y: y_mb
                                           })
                self.file_writer.add_summary(summary, self.total_runs)
            validation_error = self.validate(test_input, test_output)
            print('validation_error: {}'.format(validation_error))

    @abstractmethod
    def validate(self, data):
        raise NotImplementedError
