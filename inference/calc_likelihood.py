import numpy as np
from typing import Optional

from scipy.special import logsumexp

from inference.grids import get_default_grid, LikelihoodGrid, N_DEFAULT_SAMPLES
from inference.sampling import get_sample_pseudorandom_data, make_normalised_samples, \
    get_prior_from_samples, get_log_sample_data, get_loglikelihood_per_sample
from neutrinos.hierarchies import Hierarchy
from inference.utils import load_posterior, save_posterior


def get_likelihood_mu_sigma(data, hierarchy: Hierarchy, likeli: Optional[LikelihoodGrid] = None):
    """ Span mu sigma grid and collect mass posterior along the way. """

    if likeli is None:
        likeli = get_default_grid()

    n_mu = len(likeli.muArray)
    bin_edges = likeli.mass_log_bins

    for i, mu in enumerate(likeli.muArray):
        for j, sigma in enumerate(likeli.sigmaArray):
            logsamples = get_log_sample_data(mu, sigma, likeli.n_samples)
            samples = np.exp(logsamples)

            loglikeli = get_loglikelihood_per_sample(samples, data, hierarchy)
            weights = np.exp(loglikeli)

            for n in range(3):
                samples_posterior, _ = np.histogram(logsamples[:, n], bin_edges, weights=weights)
                likeli.mass_posterior[:-1, n] += samples_posterior

            log_sum_of_masses = np.log(np.sum(samples, axis=1))
            sum_of_masses_posterior, _ = np.histogram(log_sum_of_masses, bin_edges, weights=weights)
            likeli.mass_posterior[:-1, 3] += sum_of_masses_posterior

            likeli.loglikelihood[i, j] = logsumexp(loglikeli) - np.log(likeli.n_samples)

        print('Completed mean mass', i, ' of ', n_mu)

    # Normalise the posterior masses
    likeli.mass_posterior /= np.sum(likeli.mass_posterior, axis=0)

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
