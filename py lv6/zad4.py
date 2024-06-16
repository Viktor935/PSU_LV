import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.image as mpimg

image_path = 'example_grayscale.png'
image = mpimg.imread(image_path)

plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
plt.title('Originalna slika')
plt.axis('off')
plt.show()

X = image.reshape(-1, 1)

def quantize_image(X, n_clusters):
    k_means = KMeans(n_clusters=n_clusters, n_init=1)
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    image_compressed = np.choose(labels, values)
    image_compressed.shape = image.shape
    return image_compressed

clusters = [2, 5, 10, 20]
plt.figure(figsize=(12, 8))

for i, n_clusters in enumerate(clusters, 1):
    image_compressed = quantize_image(X, n_clusters)
    plt.subplot(2, 2, i)
    plt.imshow(image_compressed, cmap='gray')
    plt.title(f'Kvantizacija s {n_clusters} klastera')
    plt.axis('off')

plt.tight_layout()
plt.show()

#povećanjem broja klastera, kvantizirana verzija više nalikuje originalnoj slici