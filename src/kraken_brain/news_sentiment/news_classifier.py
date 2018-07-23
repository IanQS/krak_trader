"""
news_classifier.py
    classifies news
    
    21 July 2018 
        - Currently only tests against a small sample dataset of not-crypto news

Author: Ian Q.

Notes:

"""
import tensorflow as tf
import spacy

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
from kraken_brain.news_sentiment.utils import parse_dataset

def main(src):
    X_train, y_train, X_test, y_test = parse_dataset(src)
    all_words = X_train + X_test  # For tokenizing purposes
    NUM_TOKENS = 500
    
    
    

if __name__ == '__main__':
    DATASET = "review_polarity"
    main(DATASET)

