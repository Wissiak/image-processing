'''
This script shows the covariance matrix of the original image and the one of red and blue channels switched.
@author Patrick Wissiak
'''
import matplotlib.pyplot as plt
import numpy as np
r = [
    [255,255,255,255,255],
    [255,255,255,255,255],
    [255,255,255,255,255],
    [255,255,255,255,255],
    [255,255,255,255,255]
]
g = [
    [0,0,0,0,0],
    [0,0,255,0,0],
    [0,255,255,255,0],
    [0,0,255,0,0],
    [0,0,0,0,0]
]
b = [
    [0,0,0,0,0],
    [0,0,255,0,0],
    [0,255,255,255,0],
    [0,0,255,0,0],
    [0,0,0,0,0]
]

# mean: 255*5/25=51
# cov: (51^2 * 20 + (255-51)^2 *5) / 25

n_img = np.dstack((r,g,b))
n_img_flipped = np.dstack((b,g,r))

def rgb_cov(im):
    '''
    Assuming im a torch.Tensor of shape (H,W,3):
    '''
    im_re = im.reshape(-1, 3)
    im_re = im_re.astype(np.float64)
    im_re -= im_re.mean(0)
    # @ performs a matrix multiplication
    return 1/(im_re.shape[0]-1) * im_re.T @ im_re

cov = rgb_cov(n_img)
print(f'Covariance Matrix:')
print(cov)
cov_flipped = rgb_cov(n_img_flipped)
print(f'Covariance Matrix of B and R channels flipped:')
print(cov_flipped)

fig, ax = plt.subplots(2, 2, constrained_layout=True)
ax[0][0].set_title('RGB Image')
ax[0][0].imshow(n_img)
ax[1][0].set_title('Covariance of RGB Image')
ax[1][0].matshow(cov)
ax[0][1].set_title('BGR Image')
ax[0][1].imshow(n_img_flipped)
ax[1][1].set_title('Covariance of BGR Image')
ax[1][1].matshow(cov_flipped)
plt.show()