# Motivation

Acts as a single coordinator for all the models.

All models should be created within `_models` folder, and inherit from `base_model` which will allow easy class factory initialization.

The top level `coordinator` will act as an ensemble-r