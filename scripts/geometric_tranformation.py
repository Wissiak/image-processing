'''
This script implements the affine transform for rotation of an image. 
@author Patrick Wissiak
'''
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

# Load image
img = plt.imread("./images/letter-t.jpg")

# Tranform image to grayscale and float
if img.ndim > 2:
    img = np.mean(img, axis=2)
img = np.array(img[:,0:1000], dtype=float)
img = np.pad(img, 50)
(M, N) = img.shape

# By applying the affine transform, some values are mapped to the same pixels
# which results in left-out spots in the resulting image. To overcome this, 
# we must iterate over the target image and map the pixel positions backward. 
# In order to do that, the inverse matrix A^(-1) must be used.
def rotate(input, theta):
    A = [
        [np.cos(theta), -np.sin(theta), 0], 
        [np.sin(theta), np.cos(theta), 0], 
        [0, 0, 1]
    ]
    A_inv = inv(A)
    g = np.zeros((M, N))
    for ix in range(M):
        for iy in range(N):
            coord_s = np.matmul(A_inv, np.array([ix, iy, 1]))
            coord_s = coord_s.astype(np.int32)
            g[ix, iy] = input[np.mod(coord_s[0], M), np.mod(coord_s[1], N)]
    return g

g_30 = rotate(img, np.radians(30))
g_90 = rotate(img, np.radians(90))
g_180 = rotate(img, np.radians(180))

#%% Display Results
fig, ax = plt.subplots(2, 2, constrained_layout=True)
ax[0][0].imshow(img, cmap="gray")
ax[0][0].title.set_text("Original")
ax[0][1].imshow(g_30, cmap="gray")
ax[0][1].title.set_text("Rotated 30°")
ax[1][0].imshow(g_90, cmap="gray")
ax[1][0].title.set_text("Rotated 90°")
ax[1][1].imshow(g_180, cmap="gray")
ax[1][1].title.set_text("Rotated 180°")

plt.show()