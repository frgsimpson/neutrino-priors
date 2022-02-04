import numpy as np
from typing import Optional

from inference.grids import get_default_grid, LikelihoodGrid, N_DEFAULT_SAMPLES
from inference.sampling import get_loglikelihood_from_samples, get_sample_pseudorandom_data, make_normalised_samples, \
    get_prior_from_samples
from neutrinos.hierarchies import Hierarchy
from utils.loading import load_posterior, save_posterior


def get_likelihood_mu_sigma(data,  hierarchy: Hierarchy, likeli: Optional[LikelihoodGrid] = None):

    if likeli is None:
        likeli = get_default_grid()

    unit_sorted_samples = make_normalised_samples(likeli.n_samples)

    n_mu = len(likeli.muArray)
    for i, mu in enumerate(likeli.muArray):
        for j, sigma in enumerate(likeli.sigmaArray):
            samples = get_sample_pseudorandom_data(mu, sigma, unit_sorted_samples)
            likeli.loglikelihood[i, j] = get_loglikelihood_from_samples(samples, data, hierarchy)

        print('Completed mean mass', i, ' of ', n_mu)

    return likeli


def get_prior_on_masses(likeli: LikelihoodGrid, plot_type, bin_edges):
    """ Estimate the prior on the individual neutrino masses"""

    unit_sorted_samples = make_normalised_samples(likeli.n_samples)
    n_mu = len(likeli.muArray)

    n_mass_bins = len(bin_edges) - 1
    histogram_shape = (n_mass_bins, n_mass_bins)
    prior_histogram = np.zeros(histogram_shape)

    for i, mu in enumerate(likeli.muArray):
        for j, sigma in enumerate(likeli.sigmaArray):
            samples = get_sample_pseudorandom_data(mu, sigma, unit_sorted_samples)
            prior_histogram += get_prior_from_samples(samples, bin_edges, plot_type)

        print('Completed mean mass', i, ' of ', n_mu)

    return prior_histogram / np.sum(prior_histogram)


def get_posterior(hierarchy, data, n_samples: int = N_DEFAULT_SAMPLES):
    """ Either load the posterior from the file or calculate it from scratch. """

    try:
        posterior = load_posterior(hierarchy, data, n_samples)
    except FileNotFoundError:
        posterior = get_likelihood_mu_sigma(data, hierarchy)
        save_posterior(hierarchy, data, posterior)

    return posterior
