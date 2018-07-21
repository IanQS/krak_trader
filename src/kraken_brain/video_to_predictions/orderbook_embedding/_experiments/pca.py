"""
pca.py -> Investigation into normalization vs min-max scaling for featurization
for PCA. Assumption is that the transformations here will be useful in an AE

Author: Ian Q.

Notes:
    
"""

from kraken_brain.trader_configs import ALL_DATA
from kraken_brain.video_to_predictions.orderbook_embedding.utils import get_image_from_np, custom_scale
from sklearn.decomposition import PCA


if __name__ == '__main__':
    CURRENCY = 'XXRPZUSD'
    data = get_image_from_np(ALL_DATA, CURRENCY)
    normed_data = custom_scale(data, (data[0].shape[0], 100 * 2 * 2))
    model = PCA(n_components=20)
    X_reduce = model.fit_transform(normed_data)

    print(model.explained_variance_)
    print(model.explained_variance_ratio_.sum())