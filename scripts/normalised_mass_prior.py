""" Plot a representation of our prior on m1 m2. """
from matplotlib import pyplot as plt

from inference.calc_likelihood import get_prior_on_masses
from inference.grids import get_default_grid
import numpy as np

N_DEFAULT_SAMPLES = 10_000  # 1_000_000  # Samples per mu-sigma bin
N_MASS_BINS = 100
EPSILON = 1e-10
MIN_PLOT_VAL = -9  # How low data becomes black
LOG_MASS_SCALE = True   # Whether to show prior on log scale.
PLOT_TYPE = 'normalised'

bin_edges = np.linspace(0, 1.0, N_MASS_BINS + 1)
likeli = get_default_grid(n_mu=1_000, n_sigma=30, log_mass_spacing=LOG_MASS_SCALE)  # Need fine grained mu since it's log distributed
prior_grid = get_prior_on_masses(likeli, plot_type=PLOT_TYPE, bin_edges=bin_edges)
log_prior_grid = np.log(prior_grid + EPSILON)

im = plt.imshow(log_prior_grid, cmap="hot", vmin=MIN_PLOT_VAL, origin='lower', extent=[bin_edges[0], bin_edges[-1], bin_edges[0], bin_edges[-1]])
cbar = plt.colorbar(im)

plt.xlabel('Medium / Heavy')
plt.ylabel('Light / Heavy')

plt.savefig('./plots/' + PLOT_TYPE + '.png', dpi=300)
plt.show()
