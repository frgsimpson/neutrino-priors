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


# function[likeliVal, sigmaLikeli, massLikeli] = calc_mass_likelihoods(sampleArray, DATA)
# COLLECT_SIGMA_LIKELI = true;
# COLLECT_MASS_LIKELI = true;
#
# % NDATA = length(DATA.VALUE);
# % likeliArray = ones(1, length(sampleArray));
#
# switch
# DATA.TYPE
#
# case
# {'NH', 'IH'}
#
# likeliArray = normpdf(sampleArray.msqr1, DATA.VALUE(1), DATA.ERROR(1));
# likeliArray = likeliArray. * normpdf(sampleArray.msqr2, DATA.VALUE(2), DATA.ERROR(2));
# likeliArray = likeliArray. * normpdf(sampleArray.sum, DATA.VALUE(3), DATA.ERROR(3));
#
# otherwise
# error('not yet supported');
# end
#
# likeliVal = sum(likeliArray); % Add
# up
# all
# the
# likelihood
# values
#
# % But
# we
# may
# also
# want
# to
# see
# the
# marginal
# likelihood
# for a parameter like
# % sigma
# if (COLLECT_SIGMA_LIKELI) % Struct that has bins and summed values
# sigmaLikeli = calc_sigma_likeli(likeliArray, sampleArray.sum);
# end
#
# if (COLLECT_MASS_LIKELI)
#     massLikeli = calc_mass_likeli(likeliArray, sampleArray.masses);
# end
#
# end
#
# function
# massLikeli = calc_mass_likeli(likeliArray, massArray)
#
# [binEdges, NLIKELI_BINS] = get_mass_bins();
#
# massLikeli.likeli = zeros(3, NLIKELI_BINS);
#
# for i=1:3
# [N, edges, bin] = histcounts(massArray(i,:), binEdges);
# massLikeli.likeli(i,:) = accumarray(bin(bin > 0)
# ', likeliArray(bin>0)', [NLIKELI_BINS 1]);
# end
#
# massLikeli.mass = edges;
#
# massLikeli.likeli(isnan(massLikeli.likeli)) = 0;
# % if sum(isnan(massLikeli.likeli(: ))) > 0
# % here = 1
# % end
#
# end
#
# function
# sigmaLikeli = calc_sigma_likeli(likeliArray, sigmaArray)
#
# % RES_FACTOR = 200; % 0.005
# eV
# resolution
# [binEdges, NLIKELI_BINS] = get_sigma_bins();
#
# % [N, edges, bin] = histcounts(RES_FACTOR * sigmaArray, 'BinMethod', 'integers');
# [N, edges, bin] = histcounts(sigmaArray, binEdges);
#
# sigmaLikeli.likeli = accumarray(bin(bin > 0)
# ', likeliArray(bin>0)', [NLIKELI_BINS 1]);
# sigmaLikeli.sigma = edges;
#
# end
# % y = normpdf(x, mu, sigma)
# function
# likeliArray = calc_likeli(measurement, data, error)
#
# likeliArray = exp(-(measurement - data). ^ 2. / (2 * error. ^ 2));
#
# end
#
#
# end
#
# % Output
# structure
# likelihood.muArray = muArray;
# likelihood.sigmaArray = sigmaArray;
# likelihood.likeli = likeli;
#
# likelihood.posterior = posterior;
# likelihood.sigmaLikeli = sigmaLikeli; % contains
# subfields
# sigma and likeli
# likelihood.massLikeli = massLikeli;
#
# function
# intialise_arrays()
#
# logMUMIN = log10(params.muMIN);
# logMUMAX = log10(params.muMAX);
# logSIGMAMIN = log10(params.sigmaMIN);
# logSIGMAMAX = log10(params.sigmaMAX);
#
# muArray = logspace(logMUMIN, logMUMAX, params.Nmu);
# sigmaArray = logspace(logSIGMAMIN, logSIGMAMAX, params.Nsigma);
#
# end
# end
#
