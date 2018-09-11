'''
xg_regressor.py - implementation of gradient boosted model using the XGBoost Scikit-Learn API

Author: Bryan Q.

Notes:
    1) https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
        - Full list of available parameters, attributes and methods here
'''
from kraken_brain.linear_models._models.base_model import BaseModel
from xgboost import XGBRegressor

from kraken_brain.trader_configs import ALL_DATA
from kraken_brain.linear_models.utils import get_price_data, process_data


class XGRegressor(BaseModel):
    def __init__(self, pre_processing=None, model_args={}):
        super().__init__(self.__class__.__name__)
        self.model = XGBRegressor(**model_args)
        self.pre_processor = pre_processing

    def _train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def _predict(self, data):
        return self.model.predict(data)

    def load(self):
        self.model.load_model('xgboost.txt')
        self.trained = True

    def _save(self):
        self.model.save_model('xgboost.txt')


if __name__ == '__main__':
    model_args = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 100
    }

    model = XGRegressor(process_data, model_args)
    data = get_price_data(ALL_DATA, 'XXRPZUSD')
    if model.pre_processor is not None:
        data = model.pre_processor(data)

    model.train(*data)
