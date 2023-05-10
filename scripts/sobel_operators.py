'''
This script implements sobel operators and shows the results in x- and y-direction as well as the magnitude image.
@author Patrick Wissiak
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2

# Define the sobel operators
s_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
s_x = [[-1,0,1],[-2,0,2],[-1,0,1]]

def get_partial_derivatives(input, M, N):
    P = 2 * M
    Q = 2 * N
    d_dy = np.real(ifft2(fft2(input, (P, Q)) * fft2(s_y, (P, Q))))[1:M+1, 1:N+1]
    d_dx = np.real(ifft2(fft2(input, (P, Q)) * fft2(s_x, (P, Q))))[1:M+1, 1:N+1]

    M_xy = np.sqrt(d_dx**2 + d_dy**2)
    return d_dx, d_dy, M_xy

f = [np.zeros(7), np.zeros(7), np.ones(7), np.ones(7), np.ones(7), np.zeros(7), np.zeros(7)]
M = N = 7
f_d_dx, f_d_dy, f_M_xy = get_partial_derivatives(f, M, N)

l = plt.imread("./images/lions.jpg")
if l.ndim > 2:
    l = np.mean(l, axis=2)
l = np.array(l[:,0:1000], dtype=float)
(M, N) = l.shape
l_d_dx, l_d_dy, l_M_xy = get_partial_derivatives(l, M, N)


fig, ax = plt.subplots(2, 4, constrained_layout=True)

ax[0][0].imshow(f, cmap="gray")
ax[0][0].set_title("Original image")
ax[0][1].imshow(f_d_dx, cmap="gray")
ax[0][1].set_title("Derivative in x-direction")
ax[0][2].imshow(f_d_dy, cmap="gray")
ax[0][2].set_title("Derivative in y-direction")
ax[0][3].imshow(f_M_xy, cmap="gray")
ax[0][3].set_title("Magnitude image")

ax[1][0].imshow(l, cmap="gray")
ax[1][0].set_title("Original image")
ax[1][1].imshow(l_d_dx, cmap="gray")
ax[1][1].set_title("Derivative in x-direction")
ax[1][2].imshow(l_d_dy, cmap="gray")
ax[1][2].set_title("Derivative in y-direction")
ax[1][3].imshow(l_M_xy, cmap="gray")
ax[1][3].set_title("Magnitude image")

plt.show()