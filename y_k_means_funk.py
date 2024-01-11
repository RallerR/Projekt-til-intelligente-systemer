import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from imageio import imread
from skimage.transform import rescale
from skimage.color import rgb2lab, lab2rgb
from sklearn.metrics.pairwise import euclidean_distances
import os
from skimage.io import imsave

# Sæt arbejdsmappe
os.chdir("/Users/rasmusrieneck/Desktop/IntelligentSystemsProject/Test")


# "billed" skal være path til hvor billedet ligger og skal være .jpg
def lav_y_k_means(billed, output_folder):
    image_raw = imread(billed)

    if image_raw.ndim == 2:
        image_raw = np.stack((image_raw,) * 3, axis=-1)

    image_raw = image_raw[:, :, :3]
    image_width = 32
    image = rescale(image_raw, image_width / image_raw.shape[0], mode='reflect', channel_axis=2, anti_aliasing=True)

    X = rgb2lab(image).reshape(-1, 3)
    K = 2
    centers = np.array([X.mean(0) + (np.random.randn(3) / 10) for _ in range(K)])

    for i in range(2):
        y_kmeans = np.argmin(euclidean_distances(X, centers), axis=1)
        for j, c in enumerate(centers):
            points = X[y_kmeans == j]
            if len(points):
                centers[j] = points.mean(0)

    centers_rgb = lab2rgb(centers.reshape(1, -1, 3)).reshape(-1, 3)
    clustered_image = centers_rgb[y_kmeans].reshape(image.shape)

    # Normalize to [0, 1] and then scale to [0, 255]
    clustered_image = (clustered_image - clustered_image.min()) / (clustered_image.max() - clustered_image.min())
    clustered_image = (255 * clustered_image).astype(np.uint8)

    output_path = os.path.join(output_folder, os.path.basename(billed))
    imsave(output_path, clustered_image)

    return y_kmeans


# Example usage
output_folder = "/Users/rasmusrieneck/Desktop/IntelligentSystemsProject/Test"
print(lav_y_k_means("circle_1.jpg", output_folder))