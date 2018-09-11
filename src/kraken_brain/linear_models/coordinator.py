"""
coordinator.py -> acts as a high level "brain" for the linear models. Should follow
a similar interface for the non-linear models

Author: Ian Q.

Notes:
    TODO: optimize balance between retraining && continuing with model
        self.train_cost
"""

from kraken_brain.linear_models.registered_models import registered

class Master(object):
    def __init__(self):
        pass
