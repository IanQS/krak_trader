"""
Trained as a convolutional auto-encoder, then we remove the decoder and use the
generated

Author: Ian Q
"""
import tensorflow as tf
import numpy as np
from kraken_brain.utils import custom_scale
from kraken_brain.trader_configs import ALL_DATA, SUMMARY_PATH, CONV_INPUT_SHAPE
from kraken_brain.utils import get_image_from_np, clear_tensorboard, custom_scale
from tqdm import tqdm


class Autoencoder(object):
    def __init__(self, sess: tf.InteractiveSession, graph: tf.Graph, input_shape: tuple, batch_size: int,
                 debug: bool, lr: float = 0.1, epochs=20):
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
            self.encoder = self.__construct_encoder(self.shape)
            self.decoder = self.__construct_decoder(self.encoder)
            self.cost, self.train_op, self.validation = self._metric_construction

        ################################################
        # Construct loggers
        ################################################
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
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
    def _metric_construction(self):
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
        for i in tqdm(range(self.epochs)):

            ################################################
            # Train the model
            ################################################
            batch_length = len(orderbook_data)
            randomized = np.random.choice(batch_length, batch_length, replace=False)
            for j, mb in enumerate(range(batch_length // BATCH_SIZE)):
                total_runs += 1
                minibatch = orderbook_data[randomized[mb * BATCH_SIZE: (mb + 1) * BATCH_SIZE], :]
                summary, x = self.sess.run([self.summ, self.train_op],
                                   feed_dict={self.encoder_input: minibatch})
                self.file_writer.add_summary(summary, total_runs)

            ################################################
            # Validate on fixed number of points
            ################################################
            score = self.sess.run(self.validation, feed_dict={self.encoder_input: validation_data})

            print("Validation Error: Step {} Error: {}".format(i, score))


if __name__ == '__main__':
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    graph = tf.Graph()
    BATCH_SIZE = 256
    CURRENCY = 'XXRPZUSD'
    EPOCHS = 50
    model = Autoencoder(sess, graph, (100, 4), BATCH_SIZE, debug=True, epochs=EPOCHS)

    ################################################
    # Data processing steps
    ################################################
    data = get_image_from_np(ALL_DATA, CURRENCY)
    # Scale the data axis-wise
    data = custom_scale(data, (data[0].shape[0], 100, 2 * 2))
    validation_data, data = data[:BATCH_SIZE, :],  data[BATCH_SIZE:, :]

    # train, then validate
    model.train(data, validation_data)
