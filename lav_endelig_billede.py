import Andreas_aotu.auto_ecoder as AE
import matplotlib.pyplot as plt
import numpy as np


def lav_billed(data):
    width = int(np.sqrt(len(data)))
    height = int(len(data) / width)

    image_data = np.reshape(data, (width, height))

    # Lav et sort-hvidt billede
    plt.imshow(image_data, cmap='gray', interpolation='nearest')
    plt.axis('off')  # Slå akser fra
    plt.show()


for i in range(2,10):
    lav_billed(AE.Hele_encoder(i))
    print(i)

