'''
This script shows the result of adding the PDFs of multiple uniformly distributed random variables.
Note that the result is a normal distribution with mean=N/2 where N is the number of added random variables.
@author Patrick Wissiak
'''
import random
import matplotlib.pyplot as plt

sample_size = 1000000

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

ax1.hist([random.random() for _ in range(sample_size)])
ax1.set_title('PDF: Random Variable')
ax1.set_xlim(0, 1)

ax2.hist([random.random() + random.random() for _ in range(sample_size)])
ax2.set_title('PDF: Addition of 2 Random Variables')
ax2.set_xlim(0, 2)

ax3.hist([random.random() + random.random() + random.random() for _ in range(sample_size)])
ax3.set_title('PDF: Addition of 3 Random Variables')
ax3.set_xlim(0, 3)

plt.show()