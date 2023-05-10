'''
This script shows erroneous histogram equalization (r**2 instead of r). Every intensity greater than 16 is mapped to 255.
@author Patrick Wissiak
'''
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("./images/lions.jpg")
if img.ndim > 2:
    img = np.mean(img, axis=2)
img = np.array(img[:,0:1000], dtype=float)
(M, N) = img.shape

L = 255
intensities = range(L + 1)

rounded_image = np.rint(img).astype(np.int16)
probabilities = [np.count_nonzero(rounded_image == i) / (M * N) for i in intensities]

def get_equalized_image(input, is_squared=False):
    intensity_mapping = np.zeros(L + 1)
    for r in intensities:
        if is_squared:
            intensity_mapping[r] = L * np.sum(probabilities[0:r**2 + 1])
        else:
            intensity_mapping[r] = L * np.sum(probabilities[0:r + 1])

    g = np.zeros((M, N))
    for cx in range(M-1):
        for cy in range(N-1):
            g[cx,cy] = intensity_mapping[rounded_image[cx,cy]]
    return g, intensity_mapping

g, intensity_mapping1 = get_equalized_image(img)
g_r_squared, intensity_mapping2 = get_equalized_image(img, True)

equalized_probabilities = [np.count_nonzero(np.rint(intensity_mapping1) == i) / (L+1) for i in intensities]
erroneus_probabilities = [np.count_nonzero(np.rint(intensity_mapping2) == i) / (L+1) for i in intensities]

ht0 = np.histogram(img, bins=np.arange(L + 1), density=True)[0]
ht = np.histogram(g, bins=np.arange(L + 1), density=True)[0]
ht2 = np.histogram(g_r_squared, bins=np.arange(L + 1), density=True)[0]

fig, ax = plt.subplots(2, 3, constrained_layout=True)
ax[0][0].imshow(img, cmap="gray")
ax[0][0].title.set_text("Original")
ax[1][0].plot(range(L), ht0)
ax[0][0].title.set_text("Original CDF")
ax[0][1].imshow(g, cmap="gray", vmin=0, vmax=255)
ax[0][1].title.set_text("Normal Histogram Equalization")
ax[1][1].plot(range(L), ht)
ax[1][1].title.set_text("Histogram after equalization")
ax[0][2].imshow(g_r_squared, cmap="gray", vmin=0, vmax=255)
ax[0][2].title.set_text("Histogram Equalization with r**2 instead of r")
ax[1][2].plot(range(L),ht2)
ax[1][2].title.set_text("Histogram after erroneus equalization")

plt.show()