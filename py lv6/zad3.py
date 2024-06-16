from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def generate_data(n_samples, flagc):
    if flagc == 1:
        random_state = 365
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        
    elif flagc == 2:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
        
    elif flagc == 3:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples, centers=4, cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=random_state)

    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
        
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

n_samples = 500
flagc = 5  # 1 do 5

X = generate_data(n_samples, flagc)

methods = ['single', 'complete', 'average', 'ward']
plt.figure(figsize=(20, 10))

for i, method in enumerate(methods):
    plt.subplot(2, 2, i+1)
    Z = linkage(X, method=method)
    dendrogram(Z)
    plt.title(f'Dendrogram sa metodom: {method}')

plt.tight_layout()
plt.show()

#ovisno o metodi pristupa, i metodi generacije, možemo dobiti jasnije rezultate organizacije klastera koristeći dendograme