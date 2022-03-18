""" Plot a representation of our prior on m1 m2. """
from matplotlib import pyplot as plt

from inference.calc_likelihood import get_prior_on_masses
from inference.grids import get_default_grid
import numpy as np

N_DEFAULT_SAMPLES = 1_000_000  # Samples per mu-sigma bin
N_MASS_BINS = 100
EPSILON = 1e-10
PLOT_TYPES = ['heavylight', 'heavymedium', 'mediumlight']
LOG_PROB = False
MIN_PLOT_VAL = -10 if LOG_PROB else 0.  # How low data becomes black

# Need fine grained mu since it's log distributed
likeli = get_default_grid(n_mu=1_000, n_sigma=30, log_mass_spacing=True)
max_mu = likeli.muArray[-1]

for plot_type in PLOT_TYPES:
    bin_edges = np.linspace(0, 0.2, N_MASS_BINS + 1)
    prior_grid = get_prior_on_masses(likeli, plot_type=plot_type, bin_edges=bin_edges)
    log_prior_grid = np.log(prior_grid + EPSILON)

    prior_values = log_prior_grid if LOG_PROB else prior_grid

    im = plt.imshow(prior_values, cmap="hot", vmin=MIN_PLOT_VAL, origin='lower', extent=[bin_edges[0], bin_edges[-1], bin_edges[0], bin_edges[-1]])
    cbar = plt.colorbar(im)

    if plot_type == 'heavymedium':
        x_label = 'Heavy (eV)'
        y_label = 'Medium (eV)'
    elif plot_type == 'heavylight':
        x_label = 'Heavy (eV)'
        y_label = 'Light (eV)'
    else:
        x_label = 'Medium (eV)'
        y_label = 'Light (eV)'

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig('./plots/' + plot_type + '.png', dpi=300)
    plt.show()
