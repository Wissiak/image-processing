'''
This script implements the Laplacian spatial kernel as a convolution in the frequency domain to perform contrast enhancement. 
@author Patrick Wissiak
'''
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2

# Load image
img = plt.imread("./images/moon.jpg")

# Tranform image to grayscale and float
if img.ndim > 2:
    img = np.mean(img, axis=2)
img = np.array(img[:,0:1000], dtype=float)
(M, N) = img.shape

f = img.copy()

k_size = 3
k = [[1,1,1],[1,-8,1],[1,1,1]]

P = 2 * M
Q = 2 * N

H = fft2(k, [P, Q])
F = fft2(f, [P, Q])

g = np.real(ifft2(F * H))[:M,:N]
f_hat = f - g

fig, ax = plt.subplots(2, 2, constrained_layout=True)
ax[0][0].imshow(img, cmap="gray")
ax[0][0].title.set_text("Original")
ax[1][0].imshow(g, cmap="gray")
ax[1][0].title.set_text("After Laplacian Filtering")
#ax[1][0].phase_spectrum(F)
#ax[1][0].title.set_text("Laplacian")
ax[1][1].imshow(f_hat, cmap="gray", vmin=0, vmax=255) # Very important to clip the negative values after applying the Laplacian
ax[1][1].title.set_text("Contrast enhanced")

plt.show()