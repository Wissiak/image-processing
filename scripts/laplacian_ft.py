'''
This script implements the Laplacian filter transfer function to perform contrast enhancement. 
It also illustrates the Laplacian filter transfer functions as graphs.
@author Patrick Wissiak
'''
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2
from matplotlib import cm

# Load image
img = plt.imread("./images/moon.jpg")

# Tranform image to grayscale and float
if img.ndim > 2:
    img = np.mean(img, axis=2)
img = np.array(img[:,0:1000], dtype=float)
(M, N) = img.shape

f = img.copy()

P = M
Q = N

u = ((np.arange(P) - P/2) / P) ** 2
v = ((np.arange(Q) - Q/2) / Q) ** 2
#u = u/M
#v = v/N
uv = np.repeat(np.expand_dims(u, axis=-1), Q, axis=-1)
D_uv = uv + v
H_decentered = -4 * np.pi ** 2 * D_uv

# The Laplacian filter must be decentered
H = np.roll(H_decentered, int(-M/2), axis=0)
H = np.roll(H, int(-N/2), axis=1)
F = fft2(f)

g = np.real(ifft2(F * H))
f_hat = f - g

fig, ax = plt.subplots(2, 2, constrained_layout=True)
ax[0][0].imshow(f, cmap="gray")
ax[0][0].title.set_text("Original")
ax[0][1].imshow(H, cmap="gray")
ax[0][1].title.set_text("Laplacian")
ax[1][0].imshow(g, cmap="gray")
ax[1][0].title.set_text("After Laplacian Filtering")
ax[1][1].imshow(f_hat, cmap="gray", vmin=0, vmax=255)
ax[1][1].title.set_text("Contrast enhanced")

plt.show()

fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(2,2, 1, projection='3d')
X, Y = np.meshgrid(range(Q), range(P))
ax.set_title('Laplacian (-): -4 * pi**2 * D(u,v)**2')
ax.plot_surface(X, Y, H_decentered, linewidth=0, cmap=cm.coolwarm)

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.set_title('Same Laplacian (-) decentered')
ax.plot_surface(X, Y, H, linewidth=0, cmap=cm.coolwarm)

# Recaulate Laplacian for plus sign
H_decentered = 4 * np.pi ** 2 * D_uv
H = np.roll(H_decentered, int(-M/2), axis=0)
H = np.roll(H, int(-N/2), axis=1)
ax = fig.add_subplot(2, 2, 3, projection='3d')
X, Y = np.meshgrid(range(Q), range(P))
ax.set_title('Laplacian (+): 4 * pi**2 * D(u,v)**2')
ax.plot_surface(X, Y, H_decentered, linewidth=0, cmap=cm.coolwarm)

ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.set_title('Same Laplacian (+) decentered')
ax.plot_surface(X, Y, H, linewidth=0, cmap=cm.coolwarm)

plt.show()