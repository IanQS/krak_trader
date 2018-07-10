"""
pca.py -> Investigation into normalization vs min-max scaling for featurization
for PCA. Assumption is that the transformations here will be useful in an AE

Author: Ian Q.

Notes:
    
"""

import numpy as np
from kraken_brain.trader_configs import ALL_DATA
from kraken_brain.utils import get_image_from_np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, minmax_scale

def custom_normalizer(data):
    holder = []
    for outer in [0, 1]:  # bids or asks
        for inner in [0, 1]:  # price or vol
            data_ = data[:, :, inner, outer]
            holder.append(minmax_scale(data_))
    return np.moveaxis(np.asarray(holder), 0, -1)


if __name__ == '__main__':
    CURRENCY = 'XXRPZUSD'
    data = [get_image_from_np(ALL_DATA, CURRENCY)][0]
    #normed_data = custom_normalizer(data)
    data = normalize(data.reshape((data.shape[0], 100 * 2 * 2)))
    model = PCA(n_components=20)
    X_reduce = model.fit_transform(data)

    print(model.explained_variance_)
    print(model.explained_variance_ratio_.sum())