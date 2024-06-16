import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.image as mpimg

image_path = 'example.png'
image = mpimg.imread(image_path)

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title('Originalna slika')
plt.axis('off')
plt.show()

w, h, d = image.shape
image_array = np.reshape(image, (w * h, d))

n_colors = 10
kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(image_array)
labels = kmeans.predict(image_array)

image_quantized = kmeans.cluster_centers_[labels]
image_quantized = np.reshape(image_quantized, (w, h, d))

plt.figure(figsize=(8, 8))
plt.imshow(image_quantized)
plt.title(f'Kvantizirana slika s {n_colors} boja')
plt.axis('off')
plt.show()
