from kraken_brain.linear_models._models.base_model import BaseModel

class LinearRegression(BaseModel):
    def __init__(self, is_training, path):
        super().__init__(self.__class__.__name__, is_training, path)

