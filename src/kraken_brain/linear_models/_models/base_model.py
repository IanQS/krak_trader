from abc import ABC, abstractmethod

class BaseModel(ABC):
    """ All models MUST inherit from BaseModel, and use the provided methods

    """
    def __init__(self, name: str, is_training: bool, path: str):
        """

        :param name: Used in saving/loading of model. Use inheriting class' Name
        :param is_training: if not training, then immediately loads
        :param path: base path as in constants.py
        """
        self.name = name
        self.is_training = is_training  # opposite means we're loading from a set of weights
        self.path = path  # Path to load from

        if not self.is_training:
            self.load_weights()


    @classmethod
    def spawn(cls, *args, **kwargs):
        return cls(*args, **kwargs)


    @abstractmethod
    def predict(self):
        pass


    def save_weights(self):
        """ Save the weights after training to a certain folder

        :return:
        """

    def load_weights(self):
        """ Loads the weights from training

        :return:
        """