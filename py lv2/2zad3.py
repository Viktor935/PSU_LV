import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("tiger.png")

plt.imshow(img + 0.2)
plt.title('Posvijetljena')
plt.show()


plt.imshow(np.rot90(img, k=-1))
plt.title('Rotirana')
plt.show()


plt.imshow(np.fliplr(img))
plt.title('Zrcaljena')
plt.show()


plt.imshow(img[::10, ::10])
plt.title('Smanjena rezolucija 10 puta')
plt.show()


visina, sirina, _ = img.shape
cetvrt_sirina=sirina//4
cetvrt = np.ones_like(img)
cetvrt[:, cetvrt_sirina:2*cetvrt_sirina] = img[:, cetvrt_sirina:2*cetvrt_sirina]

plt.imshow(cetvrt)
plt.title('Cetvrt slike')
plt.show()