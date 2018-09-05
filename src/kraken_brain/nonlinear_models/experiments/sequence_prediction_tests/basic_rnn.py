import tensorflow as tf
import time

from kraken_brain.utils import next_batch


def construct_graph(n_steps, n_inputs, n_neurons, n_outputs):
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
    cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    return X, y, outputs


def construct_training_method(outputs, y):
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    return training_op, loss


if __name__ == '__main__':
    n_steps = 20
    n_inputs = 1
    n_neurons = 100
    learning_rate = 0.001
    n_iterations = 1500
    batch_size = 50
    n_outputs = 1
    predict_ahead = 10
    X, y, outputs = construct_graph(n_steps, n_inputs, n_neurons, n_outputs)
    training_op, loss = construct_training_method(outputs, y)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        start = time.time()
        for iteration in range(n_iterations):
            X_batch, y_batch = next_batch(batch_size, n_steps, predict_ahead)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % 100 == 0:
                mse, preds = sess.run([loss, outputs], feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)
                print(preds[0])
                print(X_batch[0][-1])


        print('Total time taken: {}'.format(time.time() - start))
