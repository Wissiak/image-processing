'''
This script shows how a normal distributed image can be multiplied by two (stretching) without changing its mean.
@author Patrick Wissiak
'''
import numpy as np
import matplotlib.pyplot as plt

mu = 50
sigma = 10
M = N = 50

f = np.random.normal(mu, sigma, size=(M,N))

mean = np.mean(f)

num_bins = 50

fig, ax = plt.subplots(2, 2, constrained_layout=True)
bins = np.arange(-50,150)

y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
    np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax[0][0].plot(bins, y, '--')
ax[0][0].set_xlabel('Values')
ax[0][0].set_ylabel('Probability')
ax[0][0].set_title(f'Initial image')
ax[0][0].axvline(mu, color='r')
ax[0][0].set_xlim(-50, 150)

f2 = f - mean
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
    np.exp(-0.5 * (1 / sigma * (bins))**2))
ax[0][1].plot(bins, y, '--')
ax[0][1].set_xlabel('Values')
ax[0][1].set_ylabel('Probability')
ax[0][1].set_title(f'Image minus mean')
ax[0][1].axvline(0, color='r')
ax[0][1].set_xlim(-50, 150)

f3 = f2 * 2
sigma3 = sigma * 2
y = ((1 / (np.sqrt(2 * np.pi) * sigma3)) *
    np.exp(-0.5 * (1 / sigma3 * (bins))**2))
ax[1][0].plot(bins, y, '--')
ax[1][0].set_xlabel('Values')
ax[1][0].set_ylabel('Probability')
ax[1][0].set_title(f'Multiplied image (factor 2)')
ax[1][0].axvline(0, color='r')
ax[1][0].set_xlim(-50, 150)

f4 = f3 + mean
y = ((1 / (np.sqrt(2 * np.pi) * sigma3)) *
    np.exp(-0.5 * (1 / sigma3 * (bins - mu))**2))
ax[1][1].plot(bins, y, '--')
ax[1][1].set_xlabel('Values')
ax[1][1].set_ylabel('Probability')
ax[1][1].set_title(f'Added mean to image')
ax[1][1].axvline(mu, color='r')
ax[1][1].set_xlim(-50, 150)

plt.show()