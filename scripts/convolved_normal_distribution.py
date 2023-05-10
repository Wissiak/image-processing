'''
This script illustrates the distribution of a convolved normal distributed image by the kernel [-1, 2, -1].
@author Patrick Wissiak
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

mu = 0
sigma = 1
M = N = 5000

f = np.random.normal(mu, sigma, size=(M))

mean = np.mean(f)

m = [-1, 2, -1]
g = np.real(ifft(fft(f) * fft(m, (M))))

fig, ax = plt.subplots(1, 2, constrained_layout=True)
bins = np.arange(-20,20)

y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
    np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax[0].plot(bins, y)
ax[0].set_xlabel('Values')
ax[0].set_ylabel('Probability')
ax[0].set_title(f'Original image')
ax[0].axvline(mu, color='r')
ax[0].set_xlim(-20, 20)

new_sigma = np.std(g)
new_mu = np.mean(g)
y = ((1 / (np.sqrt(2 * np.pi) * new_sigma)) *
    np.exp(-0.5 * (1 / new_sigma * (bins - new_mu))**2))
ax[1].plot(bins, y)
ax[1].set_xlabel('Values')
ax[1].set_ylabel('Probability')
ax[1].set_title(f'Image convolved with Filter')
ax[1].axvline(new_mu, color='r')
ax[1].set_xlim(-20, 20)

print(f'New mean: {new_mu}')
print(f'New standard deviation: {new_sigma}')


plt.show()