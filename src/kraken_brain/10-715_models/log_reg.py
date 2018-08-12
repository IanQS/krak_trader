'''
log_reg.py

Logistic regression

https://en.wikipedia.org/wiki/Logistic_regression#Iteratively_reweighted_least_squares_(IRLS)
'''

# import tensorflow as tf
import numpy as np
from model import Model
from sklearn.datasets import load_iris

class LogRegModel(Model):

	def __init__(self, *args, **kwargs):
		kwargs['scope'] = self.__class__.__name__
		super().__init__(*args, **kwargs)
	
	"""
	IRLS

	w^T = [b_0, b_1, ...]
	x(i) = [1, x_1(i), x_2(i), ...]^T
	u(i) = 1/(1 + exp{-w^T x(i)})

	w_{k+1} = (X^T S_k X)^-1 X^T (S_k X w_k + y - u_k)

	where
	S = diag( u(i)*(1-u(i)) ) is a diagonal weighting matrix
	u = [u(1), u(2), ...] is the vector of expected values
	X = [1 x_1(1) x_2(1) ...  the regressor matrix
	     1 x_1(2) x_2(2) ...
	     ...                ]
	y(i) = [y(1), y(2), ...]^T the vector of response variables
	"""
	def _construct_model(self, shape):
		pass
		

	def _construct_training(self, model_output):
		self.y = tf.placeholder(tf.float32, (self.X.shape[0], 1), name='Y')
		print(model_output.shape, self.y.shape)
		# self.loss = tf.losses.mean_squared_error(model_output, self.y)

			tf.losses.softmax_cross_entropy(
    onehot_labels,
    logits,
    weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
)
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

		validation_op = tf.metrics.mean_squared_error(model_output, self.y)
		tf.summary.scalar("Validation Error", validation_op[0])
		validation_op = tf.Print(validation_op, [validation_op], message='validation: ')

		return (cost, train_operation, validation_op)

	def validate(self, model_output):  #data):
		"""
		calculate loss based on current model after n epochs
		"""
		
		indices = np.random.choice(np.arange(len(test_input_data)),
							   BATCH_SIZE, replace=False)

		test_batch_input = test_input_data[indices]
		test_batch_output = test_output_data[indices]

		validation_score = self.sess.run([self.validation_op],
                                           feed_dict={
                                               self.X: test_batch_input,
                                               self.y: test_batch_output
                                           })
		res = self.sess.run([self.model],
							feed_dict={
								self.X: test_batch_input
							})

		self.variable_summaries(test_batch_output - res)
		return validation_score

if __name__ == '__main__':
	X, y = load_iris(True)
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
	
	shape = (BATCH_SIZE, X.shape[1])  # X.shape
	model = LinRegModel(session, graph, shape, summary_path='./events/', epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
	model.train(X_train, y_train, X_test, y_test)
	validation_error = model.validate(X_test, y_test)
	print('validation_error: {}'.format(validation_error))