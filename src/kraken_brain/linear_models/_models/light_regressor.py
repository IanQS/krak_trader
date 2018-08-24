"""
lightgbm.py - implementation of gradient boosted decision tree using the lightgbm API by Microsoft.

Author: Bryan Q.

Notes:
    1) https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/sklearn_example.py
        - Implementation is based off this
    2) https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api
        - Full list of available parameters here
"""
from kraken_brain.linear_models._models.base_model import BaseModel
from lightgbm import LGBMRegressor

from kraken_brain.trader_configs import ALL_DATA
from kraken_brain.linear_models.utils import get_price_data, process_data

import numpy as np


class LightRegressor(BaseModel):
    def __init__(self, pre_processing=None):
        super().__init__(self.__class__.__name__)
        self.model = LGBMRegressor(objective='regression',
                                   num_leaves=10,
                                   learning_rate=0.03,
                                   importance_type='split')
        self.pre_processor = pre_processing

    def _train(self, train_x, train_y):
        self.model.fit(train_x,
                       train_y,
                       eval_metric='l2',
                       early_stopping_rounds=5)

    def _predict(self, data):
        return self.model.predict(data, num_iteration=self.model.best_iteration_)

    def load(self):
        # to load the weights
        pass

    def _save(self):
        '''
        Notes:
            1) https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMRegressor
               https://lightgbm.readthedocs.io/en/latest/Parameters.html#weight_column
                - list of possible weights
        '''
        variables = {'best_score_': self.best_score_,
                     'feature_importances_': self.feature_importances_}
        np.savez(self.weights_path, **variables)


if __name__ == '__main__':
    model = LightRegressor(process_data)
    data = get_price_data(ALL_DATA, 'XXRPZUSD')
    if model.pre_processor is not None:
        data = model.pre_processor(data)

    model.train(*data)
