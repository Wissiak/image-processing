'''
This script shows the distribution of squared normal distributed values. 
Note that the resulting mean is the standard deviation squared.
@author Patrick Wissiak
'''
import numpy as np
import matplotlib.pyplot as plt

mu = 0
sigma = 10
M = N = 300

f = np.random.normal(mu, sigma, size=(M,N))

mean = np.mean(f)
print(f'Mean: {mean}')
print(f'Std: {sigma}')

f_squared = f**2
mean_squared = np.mean(f_squared)
std_squared = np.std(f_squared)
print(f'Mean (squared): {mean_squared}')
print(f'Std (squared): {std_squared}')

num_bins = 50

fig, ax = plt.subplots(2, 1, constrained_layout=True)

def plot_hist(axis, img, sigma, mu, title):
    n, bins, patches = axis.hist(img, num_bins, density=True)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
        np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    axis.plot(bins, y, '--')
    axis.set_xlabel('Values')
    axis.set_ylabel('Probability')
    axis.set_title(f'{title} with sigma={sigma} and mean={mu}')
    axis.axvline(mu, color='r')

plot_hist(ax[0], f, sigma, mu, 'Normal distributed random values')
plot_hist(ax[1], f_squared, std_squared, mean_squared, 'Squared normal distributed random values')

plt.show()