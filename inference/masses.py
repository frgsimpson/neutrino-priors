""" Routines relating to the evaluation of individual neutrino masses. """
from typing import Optional

import numpy as np
from scipy.special import logsumexp

from inference.grids import LikelihoodGrid, get_default_grid
from inference.sampling import make_normalised_samples, get_sample_pseudorandom_data, get_loglikelihood_from_samples, \
    get_loglikelihood_per_sample, get_sample_data, get_log_sample_data
from neutrinos.constraints import NeutrinoConstraint
from neutrinos.hierarchies import Hierarchy

MAX_MASS = 0.08  # eV


def get_mass_bin_edges(n_edges: int, log_spacing: bool = False):
    """ Edges of bins used for mass posterior """

    if log_spacing:
        MIN_MASS = 0.002
        edges = np.logspace(np.log10(MIN_MASS), np.log10(MAX_MASS), n_edges)
    else:
        edges = np.linspace(0, MAX_MASS, n_edges)

    return edges


def get_posterior_on_masses(data,  hierarchy: Hierarchy, likeli: Optional[LikelihoodGrid] = None):
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
                likeli.mass_posterior[:, n] += samples_posterior

            log_sum_of_masses = np.log(np.sum(samples, axis=1))
            sum_of_masses_posterior, _ = np.histogram(log_sum_of_masses, bin_edges, weights=weights)
            likeli.mass_posterior[:, 3] += sum_of_masses_posterior

            likeli.loglikelihood[i, j] = logsumexp(loglikeli) - np.log(likeli.n_samples)

        print('Completed mean mass', i, ' of ', n_mu)

    return likeli


def get_posterior_on_masses(likeli: LikelihoodGrid, data: NeutrinoConstraint, hierarchy: Hierarchy, n_mass_bins: int = 100):
    """ Estimate the prior on the individual neutrino masses"""

    n_mu = len(likeli.muArray)

    bin_edges = get_mass_bin_edges(n_edges=n_mass_bins + 1)
    n_mass_bins = len(bin_edges) - 1
    posterior_array = np.zeros((n_mass_bins, 3))

    for i, mu in enumerate(likeli.muArray):
        for j, sigma in enumerate(likeli.sigmaArray):
            samples = get_sample_data(mu, sigma, likeli.n_samples)
            loglikeli = get_loglikelihood_per_sample(samples, data, hierarchy)
            weights = np.exp(loglikeli)
            for n in range(3):
                samples_posterior, _ = np.histogram(samples[:, n], bin_edges, weights=weights)
                posterior_array[:, n] += samples_posterior

        print('Completed mean mass', i, ' of ', n_mu)

    posterior_array /= np.sum(posterior_array)

    return bin_edges[:-1], posterior_array
