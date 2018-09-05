import time
from abc import ABC, abstractmethod
from constants import MODEL_WEIGHTS_PATH
from sklearn.metrics import mean_squared_error


class BaseModel(ABC):
    """ All models MUST inherit from BaseModel, and use the provided methods

    """
    def __init__(self, name: str):
        self.name = name
        self.weights_path = MODEL_WEIGHTS_PATH.format(self.name)
        self.model = None
        self.trained = False

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
        # Metadata for controller
        ################################################
        self.train_cost = time.time() - start
        self.training_error = mean_squared_error(train_y, self.predict(train_x))
        print('{}: Trained in {} s, MSE of {}'.format(
            self.name, self.train_cost, self.training_error)
        )


        ################################################
        # Print out validation score if wanted
        ################################################
        if test_error is not None:  # FEED COST FUNCTION
            y_pred = self.predict(test_x)
            error = test_error(test_y, y_pred)
            print('{} validation-score {}'.format(self.name, error))

    @abstractmethod
    def _train(self, train_x, train_y):
        raise NotImplementedError

    def predict(self, data):
        if not self.trained:
            raise Exception('Predict called on untrained {}'.format(self.name))
        return self._predict(data)

    @abstractmethod
    def _predict(self, data):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        """ Load in the weights

        :return:
        """
        raise NotImplementedError

    def save(self):
        """ Save weights

        :return:
        """
        if not self.trained:
            raise Exception('Predict called on untrained {}'.format(self.name))
        self._save()

    @abstractmethod
    def _save(self):
        raise NotImplementedError
