import torch
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
from optical_aberration import zernike_polynomial_generator

# n (order) ranges from 0 to n_range
def plot_zernikes(n_range=5, resolution=30):

    figs, axes = plt.subplots(n_range+1, n_range*2+1, subplot_kw={'projection':'polar'})
    plt.rcParams["figure.figsize"] = (10, 10)

    for row in range(n_range+1):
        for col in range(2*n_range+1):
            axes[row][col].set_axis_off()

    zernike_generator = zernike_polynomial_generator()
    for n in range(n_range+1):
        for m in range(-n, n+1, 2):
            rs, ts, Z_polar = zernike_generator.generate_zernike_polar(n, m, 20)

            axes[n][m + n_range].contourf(ts, rs, Z_polar, cmap='Blues')
            axes[n][m + n_range].title.set_text('n={}, m={}'.format(n, m))
    plt.show()

plot_zernikes(6)
