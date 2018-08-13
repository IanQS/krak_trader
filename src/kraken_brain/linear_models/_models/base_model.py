import time
from abc import ABC, abstractmethod
from constants import MODEL_WEIGHTS_PATH
from sklearn.metrics import mean_squared_error

class BaseModel(ABC):
    """ All models MUST inherit from BaseModel, and use the provided methods

    """
    def __init__(self, name: str, is_training: bool):
        """

        :param name: Used in saving/loading of model. Use inheriting class' Name
        :param is_training: if not training, then immediately loads
        :param path: base path as in constants.py
        """
        self.name = name
        self.weights_path = MODEL_WEIGHTS_PATH.format(self.name)
        self.model = None

    def train(self, train_x, train_y, test_x, test_y, test_error=None):
        """ Wrapper around training/ validation for displaying

        :param model:
        :param x:
        :param y:
        :param is_train:
        :return:
        """

        ################################################
        # Train, and get "cost" of training (training time)
        ################################################
        start = time.time()
        self._train(train_x, train_y)

        # Parameters below are for optimization by the coordinator
        self.train_cost = time.time() - start
        self.training_error = mean_squared_error(train_y, self.predict(train_x))
        print('{}: Trained in {} s, MSE of {}'.format(
            self.name, self.train_cost, self.training_error)
        )

        ################################################
        # Print out 
        ################################################
        if test_error is not None:  # FEED COST FUNCTION
            y_pred = self.predict(test_x)
            error = test_error(test_y, y_pred)
            print('{} validation-score {}'.format(self.name, error))


    @abstractmethod
    def _train(self, train_x, train_y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        """ Load in the weights

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def save(self):
        """ Save weights

        :return:
        """
        raise NotImplementedError