"""
linear_regression.py - acts as a reference for:
    1) loading the base model, and filling in the abstract methods
    2) loading the data (for local testing)

Author: Ian Q.

Notes:
    1) Coordinator.py handles actual running of all registered models
    2) This is not suitable for time series, just acts as a reference
    3) http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        - we get coef_ and intercept_ bc sklearn Linear Regression doesn't have native saving
"""

# Model things - base class, and sklearn model to be wrapped
from kraken_brain.linear_models._models.base_model import BaseModel
from sklearn.linear_model import LinearRegression as lin_reg

# Get price data
from kraken_brain.trader_configs import ALL_DATA
from kraken_brain.linear_models.utils import get_price_data, process_data

import numpy as np


class LinearRegression(BaseModel):
    def __init__(self, pre_processing=None):
        super().__init__(self.__class__.__name__)
        self.model = lin_reg()
        self.pre_processor = pre_processing

    def _train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def _predict(self, data):
        """ Data has same shape as train_X (except for minibatch size, obv)

        :param data:
        :return:
        """
        return self.model.predict(data)

    def load(self):
        vars_ = np.load('/tmp/trash_weights.npz')
        for k, v in vars_.items():
            setattr(self.model, k, v)
        self.trained = True

    def _save(self):
        variables = {'coef_': self.model.coef_,
                     'intercept_': self.model.intercept_}
        np.savez(self.weights_path, **variables)


if __name__ == '__main__':
    model = LinearRegression(process_data)
    data = get_price_data(ALL_DATA, 'XXRPZUSD')
    if model.pre_processor is not None:
        data = model.pre_processor(data)

    # construct pre_processor such that it returns in the format appropriate
    # for doing *data
    model.train(*data)
