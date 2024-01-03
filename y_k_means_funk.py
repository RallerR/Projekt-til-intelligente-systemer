import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from imageio import imread
from skimage.transform import rescale
from skimage.color import rgb2lab, lab2rgb
from sklearn.metrics.pairwise import euclidean_distances
import os

# Sæt arbejdsmappe
os.chdir("/Users/andreasantoft/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/J24/billeder")


# "billed" skal være path til hvor billedet ligger og skal være .jpg
def lav_y_k_means(billed):
    image_raw = imread(billed)[:, :, :3]
    image_width = 10
    image = rescale(image_raw, image_width / image_raw.shape[0], mode='reflect', channel_axis=2, anti_aliasing=True)

    # Konverter billedet til LAB farverum
    X = rgb2lab(image).reshape(-1, 3)

    # K-means klyngeanalyse
    K = 2
    centers = np.array([X.mean(0) + (np.random.randn(3) / 10) for _ in range(K)])

    # Gentag estimation et antal gange
    for i in range(2):
        y_kmeans = np.argmin(euclidean_distances(X, centers), axis=1)

        # Flyt centrene til middelværdien af deres tildelte punkter
        for j, c in enumerate(centers):
            points = X[y_kmeans == j]
            if len(points):
                centers[j] = points.mean(0)

        return y_kmeans

print(lav_y_k_means("2.jpg"))