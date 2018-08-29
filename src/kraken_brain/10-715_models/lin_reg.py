'''
lin_reg.py

Linear regression
'''

import tensorflow as tf
import numpy as np
from model import Model
from sklearn.datasets import load_boston

NUM_EPOCHS = 10

class LinRegModel(Model):

	def __init__(self, *args, **kwargs):
		kwargs['scope'] = self.__class__.__name__
		super().__init__(*args, **kwargs)
		
	# Y = XW + b
	# X is nxm, W is mx1, b is nx1
	def _construct_model(self, shape):
		self.X = tf.placeholder(tf.float32, shape, name='X')
		W = tf.get_variable(name='W', dtype=tf.float32,
							shape=(self.X.shape[1], 1))
		# print("W", X.shape[1])
		b = tf.get_variable(name='b', dtype=tf.float32,
							shape=(self.X.shape[0], 1))
		# print("b", X.shape[0])

		print('_construct_model:')
		print('  X: ', self.X.shape)
		print('  W: ', (self.X.shape[1], 1))
		print('  b: ', (self.X.shape[0], 1))
		y_pred = tf.matmul(self.X, W) + b

		return y_pred

	def _construct_training(self, model_output):
		self.y = tf.placeholder(tf.float32, (self.X.shape[0], 1), name='Y')
		print(model_output.shape, self.y.shape)
		self.loss = tf.losses.mean_squared_error(model_output, self.y)
		cost = tf.reduce_mean(self.loss, name='model_cost')
		cost = tf.Print(cost, [cost], message='cost: ')
		tf.summary.scalar("Cost", cost)
		# print('Cost: {}'.format(cost))

		# kraken stuff-------------------
		optimizer = tf.train.AdamOptimizer(self.lr)

		grads = optimizer.compute_gradients(cost)
		# Update the weights wrt to the gradient
		train_operation = optimizer.apply_gradients(grads)
		# Save the grads with tf.summary.histogram
		for index, grad in enumerate(grads):
			tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])
		# kraken stuff-----------------------

		# train_operation = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(cost)
		
		# validation = tf.metrics.mean_squared_error(model_output, self.y)
		# tf.summary.scalar("Validation Error", validation[0])

		# validation = self.validate(model_output)
		validation_op = tf.metrics.mean_squared_error(model_output, self.y)
		tf.summary.scalar("Validation Error", validation_op[0])
		# print('Validation error: {}'.format(validation))
		validation_op = tf.Print(validation_op, [validation_op], message='validation: ')

		return (cost, train_operation, validation_op)

	def validate(self, test_input_data, test_output_data):  #data):
		"""
		calculate loss based on current model after n epochs
		"""
		
		indices = np.random.choice(np.arange(len(test_input_data)),
							   BATCH_SIZE, replace=False)

		test_batch_input = test_input_data[indices]
		test_batch_output = test_output_data[indices]

		mse = self.sess.run([self.validation_op],
                                           feed_dict={
                                               self.X: test_batch_input,
                                               self.y: test_batch_output
                                           })
		res = self.sess.run([self.model],
							feed_dict={
								self.X: test_batch_input
							})
		# self.variable_summaries(self.y[indices] - test_batch_output)
		self.variable_summaries(test_batch_output - res)

        # self.file_writer.add_summary(mse, self.total_runs + 1)
		# return tf.metrics.mean_squared_error(model_output, self.y)
		return mse

if __name__ == '__main__':
	X, y = load_boston(True)
	X = np.asarray(X, np.float32)
	y = np.asarray(y, np.float32)
	y = np.expand_dims(y, axis=-1)

	train_test_ratio = 0.7
	train_test_split = round(train_test_ratio * len(X))

	BATCH_SIZE = len(X) - train_test_split # 30 # config file plz

	indices = np.random.choice(np.arange(len(X)),
							   len(X), replace=False)

	train_indices = indices[:train_test_split]
	test_indices = indices[train_test_split:]

	X_train = X[train_indices]
	X_test = X[test_indices]
	y_train = y[train_indices]
	y_test = y[test_indices]

	session = tf.InteractiveSession()
	graph = tf.Graph()
	
	shape = (BATCH_SIZE, 13)  # X.shape

	# shape = X.shape
	model = LinRegModel(session, graph, shape, summary_path='./events/', epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
	# X = X.transpose()
	# print('before model training: X:%s, y:%s'%(X.shape,y.shape))
	model.train(X_train, y_train, X_test, y_test)
	# print('*' * 30)
	# print(dir(model))
	# print('*' * 30)
	validation_error = model.validate(X_test, y_test)
	print('validation_error: {}'.format(validation_error))
