"""
Builds basic_rnn.py from _experiments. Uses a sliding window in predictions


Author: Ian Q.

Notes:

"""
import time

import tensorflow as tf
from kraken_brain.linear_models._models.base_model import BaseModel
from kraken_brain.utils import toy_data_generator

class RNN_Regressor(BaseModel):
    def __init__(self, pre_processing=None):
        """Only construct the model in _train - this way if we load in late,
        no worries

        :param pre_processing:
        """
        super().__init__(self.__class__.__name__)
        self.sess = tf.Session()
        self.pre_processor = pre_processing

        self.n_steps = 20
        self.n_inputs = 1  # Dimensionality of input
        self.n_outputs = 1  # Dimensionality of output
        self.n_neurons = 100  # Fixed neuron number for all layers
        self.learning_rate = 0.001
        self.n_iterations = 1500
        self.batch_size = 50
        self.predict_ahead = 2



    def _construct_graph(self):
        X = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        y = tf.placeholder(tf.float32, [None, self.n_steps, self.n_outputs])
        cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=tf.nn.relu)
        rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, self.n_outputs)
        outputs = tf.reshape(stacked_outputs, [-1, self.n_steps, self.n_outputs])
        return X, y, outputs

    def _construct_training_method(self, outputs, y,):
        loss = tf.reduce_mean(tf.square(outputs - y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)
        return training_op, loss

    def train(self, train_x, train_y, test_x, test_y, test_error=None):
        """ Wrapper around training/ validation for displaying
        """

        ################################################
        # Train
        ################################################
        start = time.time()
        self._train(train_x, train_y)
        self.trained = True



        ################################################
        # Print out validation score if wanted
        ################################################
        if test_error is not None:  # FEED COST FUNCTION
            y_pred = self.predict(test_x)
            error = test_error(test_y, y_pred)
            print('{} validation-score {}'.format(self.name, error))

    def _train(self, train_x, train_y):
        self.X, self.y, self.model = self._construct_graph()
        self.optimizer, self.loss = self._construct_training_method(self.model, self.y)
        self.sess.run(tf.global_variables_initializer())
        for iteration in range(self.n_iterations):
            X_batch, y_batch = toy_data_generator(self.batch_size, self.n_steps, self.predict_ahead)
            self.sess.run(self.optimizer, feed_dict={self.X: X_batch, self.y: y_batch})
            if iteration % 100 == 0:
                mse, preds = self.sess.run([self.loss, self.model], feed_dict={self.X: X_batch, self.y: y_batch})
                print(iteration, "\tMSE:", mse)
                print(preds[0])
                print(X_batch[0][-1])

    def _predict(self, data):
        """ Data has same shape as train_X (except for minibatch size, obv)

        :param data:
        :return:
        """
        return self.sess.run([self.model], feed_dict = {self.X: data})

    def load(self):
        pass

    def _save(self):
        pass

if __name__ == '__main__':
    test_class = RNN_Regressor()
    test_class.train(None, None, None, None, None)