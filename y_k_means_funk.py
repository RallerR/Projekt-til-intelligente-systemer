import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from imageio import imread
from skimage.transform import rescale
from skimage.color import rgb2lab, lab2rgb
from sklearn.metrics.pairwise import euclidean_distances
import os
import torch

# Sæt arbejdsmappe
os.chdir("/Users/andreasantoft/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/J24/billeder")
np.random.seed(0)

# "billed" skal være path til hvor billedet ligger og skal være .jpg
def lav_y_k_means(billed):
    image_raw = imread(billed)[:, :, :3]
    image_width = 100
    image = rescale(image_raw, image_width / image_raw.shape[0], mode='reflect', channel_axis=2, anti_aliasing=True)

    # Konverter billedet til LAB farverum
    X = rgb2lab(image).reshape(-1, 3)

    # K-means klyngeanalyse
    K = 2
    centers = np.array([X.mean(0) + (np.random.randn(3) / 10) for _ in range(K)])

    # i er ikke brugt ???
    for i in range(10):
        y_kmeans = np.argmin(euclidean_distances(X, centers), axis=1)

        # c er ikke brugt ??
        for j, c in enumerate(centers):
            points = X[y_kmeans == j]
            if len(points):
                centers[j] = points.mean(0)
        
    if y_kmeans[0] == 1:
        y_kmeans[y_kmeans == 1] = 2
        y_kmeans[y_kmeans == 0] = 1
        y_kmeans[y_kmeans == 2] = 0
    
    y_kmeans_tensor = torch.tensor(y_kmeans, dtype=torch.float32)

    return y_kmeans_tensor

