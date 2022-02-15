""" View the likelihood for m1 masses while fixing m2 and m3. """

import numpy as np
from matplotlib import pyplot as plt

from inference.sampling import get_loglikelihood_per_sample
from neutrinos.constraints import load_pure_sum_masses, load_pure_splittings
from neutrinos.hierarchies import Hierarchy

MIN_M1 = 1e-3  # Smallest plotted mass
MAX_M2 = 0.5  # Largest plotted mass
N_GRID = 10_000
M2_MASS = 5e-2
M3_MASS = 1e-4
SIGMA_OME_SIGMA = 0.05

m1_vals = np.logspace(np.log10(MIN_M1), np.log10(MAX_M2), N_GRID)


def prepare_mass_samples():
    samples = np.zeros((N_GRID, 3))
    samples[:, 0] = M3_MASS
    samples[:, 1] = M2_MASS
    samples[:, 2] = m1_vals
    return np.sort(samples, axis=1)

# Collect data
cosmo_data = load_pure_sum_masses()
splitting_data = load_pure_splittings()
samples = prepare_mass_samples()

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(16, 8))

for (ax,  hier) in zip(axes.flat, Hierarchy):

    # Calculate likelihoods
    m1_cosmo_log_likelihood = get_loglikelihood_per_sample(samples, cosmo_data, hier)
    m1_cosmo_likelihood = np.exp(m1_cosmo_log_likelihood)

    m1_split_log_likelihood = get_loglikelihood_per_sample(samples, splitting_data, hier)
    m1_split_likelihood = np.exp(m1_split_log_likelihood)

    # Normalise for plotting
    normalisation = np.max(m1_split_likelihood) / np.max(m1_cosmo_likelihood)
    m1_cosmo_likelihood *= normalisation

    #  Plot
    ax.semilogx(m1_vals, m1_split_likelihood)
    ax.semilogx(m1_vals, m1_cosmo_likelihood, '--')

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('$m_1$ / eV')
plt.grid(False)

plt.savefig('fig1', dpi=300)
plt.show()
