'''
This script shows the wraparound-error of convolution in the frequency domain with a kernel of size (11, 11).
@author Patrick Wissiak
'''
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2

M = 200
N = 200

# Generate image
f = np.full((M // 2, N), 255)
f = np.pad(f, ((M // 2, 0),(0, 0)), 'constant', constant_values=128)

k_size = 11
k = np.ones((k_size, k_size))

H = fft2(k, [M, N])
F = fft2(f, [M, N])

# Convolve with a mean filter to show the wraparound error
# Note that the black points do not contribute to the result
# of the convolution and therefore do not affect wraparound error
g = np.real(ifft2(F * H))

f2 = np.full((M // 2, N), 255)
f2 = np.pad(f2, ((M // 2, 0),(0, 0)), 'constant', constant_values=128)
# Simulate periodic function
#f2 = np.hstack((f2, np.fliplr(f2)))
#f2 = np.vstack((f2, f2))
#f2 = np.pad(f2, (M, N), 'reflect')
k_size2 = 11
k2 = np.ones((k_size2, k_size2))
(M2, N2) = f2.shape
P = M2 + k_size2 - 1
Q = N2 + k_size2 - 1
H2 = fft2(k2, [P, Q])
F2 = fft2(f2, [P, Q])
# Gets clipped -k_size2 on both axes
g2 = np.real(ifft2(F2 * H2))

g3 = g2[k_size2:M, k_size2:N]

fig, ax = plt.subplots(2, 3, constrained_layout=True)
ax[0][0].imshow(f, cmap="gray")
ax[0][0].title.set_text(f"Original 1, shape={f.shape}")

ax[0][1].imshow(g, cmap="gray", vmin=0, vmax=np.max(g))
ax[0][1].title.set_text(f"Wraparound after Filter, shape={g.shape}")

ax[0][2].axis('off')

ax[1][0].imshow(f2, cmap="gray", vmin=0, vmax=255)
ax[1][0].title.set_text(f"Original 2, shape={f2.shape}")

ax[1][1].imshow(g2, cmap="gray", vmin=0, vmax=np.max(g2))
ax[1][1].title.set_text(f"With Padding = A + B - 1, shape={g2.shape}")

ax[1][2].imshow(g3, cmap="gray", vmin=0, vmax=np.max(g2))
ax[1][2].title.set_text(f"Removed Padding, shape={g3.shape}")

plt.show()