"""
light_regressor.py - implementation of gradient boosted model using the LightGBM Scikit-Learn API

Author: Bryan Q.

Notes:
    1) https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/sklearn_example.py
        - Implementation is based off this
    2) https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api
        - Full list of available parameters here
"""
from kraken_brain.linear_models._models.base_model import BaseModel
from lightgbm import LGBMRegressor
from sklearn.externals import joblib

from kraken_brain.trader_configs import ALL_DATA
from kraken_brain.linear_models.utils import get_price_data, process_data


class LightRegressor(BaseModel):
    def __init__(self, pre_processing=None, model_args={}):
        super().__init__(self.__class__.__name__)
        self.model = LGBMRegressor(**model_args)
        self.pre_processor = pre_processing

    def _train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def _predict(self, data):
        return self.model.predict(data, num_iteration=self.model.best_iteration_)

    def load(self):
        '''
        Load the last saved model.

        Notes:
        1) https://github.com/Microsoft/LightGBM/issues/1217#issuecomment-360352312
            - Since we are using sklearn interface we cannot use the regular save/load provided by lightgbm
        '''
        self.model = joblib.load('lightgbm.pkl')
        self.trained = True

    def _save(self):
        '''
        Save the model for later use.

        Notes:
        1) https://github.com/Microsoft/LightGBM/issues/1217#issuecomment-360352312
            - Since we are using sklearn interface we cannot use the regular save/load provided by lightgbm
        '''
        joblib.dump(self.model, 'lightgbm.pkl')


if __name__ == '__main__':
    model_args = {
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'importance_type': 'split',
        'n_iterations': 20
    }

    model = LightRegressor(process_data, model_args)
    data = get_price_data(ALL_DATA, 'XXRPZUSD')
    if model.pre_processor is not None:
        data = model.pre_processor(data)

    model.train(*data)
