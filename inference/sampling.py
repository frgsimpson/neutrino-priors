from scipy.special import logsumexp
import numpy as np

from neutrinos.constraints import NeutrinoConstraint
from neutrinos.hierarchies import Hierarchy


def make_normalised_samples(n_samples: int):
    """ Create a base set of random samples drawn from a Gaussian prior to represent the three neutrino masses. """

    sample_shape = (n_samples, 3)
    unit_sorted_samples = np.random.normal(loc=0., scale=1., size=sample_shape)
    unit_sorted_samples.sort(axis=1)
    return unit_sorted_samples


def get_sample_pseudorandom_data(mu, sigma, unit_sorted_samples):
    """ Returns samples from lognormal with params mu and sigma.
    Mu is median value in eV.Draw samples from prior using log mu
    Seeded by draws from a unit normal distribution to avoid repeated computation.
    """

    log_mass_samples = unit_sorted_samples * sigma + np.log(mu)

    return np.exp(log_mass_samples)


def get_log_sample_data(mu, sigma, N_SAMPLES):
    """ Returns log samples from lognormal with params mu and sigma.
    Mu is median value in eV.Draw samples from prior using log mu
    """

    sample_shape = (N_SAMPLES, 3)
    logSample = np.random.normal(loc=np.log(mu), scale=sigma, size=sample_shape)

    logSample.sort(axis=1)   # Put masses in order
    return logSample


def get_sample_data(mu, sigma, N_SAMPLES):
    """ Returns samples from lognormal with params mu and sigma.
    Mu is median value in eV.Draw samples from prior using log mu
    """

    logSample = get_log_sample_data(mu, sigma, N_SAMPLES)
    return np.exp(logSample)


def get_loglikelihood_from_samples(sample, data: NeutrinoConstraint, hierarchy: Hierarchy):
    """ Monte carlo estimate of p(data | mu, sigma) Perhaps use nested sampling? Can be awkward for hyperpriors
     Space is (mu, sigma, m1, m2, m3) and pretty narrow permitted region. But NS might work.  """

    loglikeli_array = get_loglikelihood_per_sample(sample, data, hierarchy)
    n_samples = sample.shape[0]

    all_likeli = logsumexp(loglikeli_array)

    return all_likeli - np.log(n_samples)  # p(D|M) ~ 1/N sum_samples p(D|M, sample) so log p is logsumexp(liklei) - log N


def get_loglikelihood_per_sample(sample, data: NeutrinoConstraint, hierarchy: Hierarchy):

    if hierarchy == Hierarchy.Normal:
        msqr1 = sample[:, 1] ** 2 - sample[:, 0] ** 2  # M ^ 2 - S ^ 2 ie m2 - m1
        msqr2 = sample[:, 2] ** 2 - sample[:, 1] ** 2  # L ^ 2 - M ^ 2 ie m3 - m2
    else:
        msqr1 = sample[:, 2] ** 2 - sample[:, 1] ** 2  # L ^ 2 - S ^ 2 % m2 - m3
        msqr2 = sample[:, 2] ** 2 - sample[:, 0] ** 2  # L ^ 2 - M ^ 2 % m2 - m1

    mass_sum = np.sum(sample, axis=1)

    loglikeli_array = log_pdf(msqr1, data.m21_sqr, data.m21_sqr_error)   # Smaller mass gap
    loglikeli_array += log_pdf(msqr2, data.m31_sqr, data.m31_sqr_error)  # Larger mass gap
    loglikeli_array += log_pdf(mass_sum, data.sum_of_masses_offset, data.sum_of_masses_one_sigma)

    return loglikeli_array


def get_prior_from_samples(samples, bin_edges, plot_type):
    """ Counts the samples in each mass bin. """

    if plot_type == 'heavymedium':
        m1 = samples[:, 1]
        m2 = samples[:, 2]
    elif plot_type == 'heavylight':
        m1 = samples[:, 0]
        m2 = samples[:, 2]
    elif plot_type == 'mediumlight':
        m1 = samples[:, 0]
        m2 = samples[:, 1]
    elif plot_type == 'normalised':
        m1 = samples[:, 0] / samples[:, 2]
        m2 = samples[:, 1] / samples[:, 2]
    else:
        raise NotImplementedError('Unknown plot type', plot_type)

    hist, m1edges, m2edges = np.histogram2d(m1, m2, bins=bin_edges)
    return hist


def log_pdf(measurement, data, error) -> np.ndarray:
    """ Faster than using official logpdf routines. """
    return -0.5*((measurement - data)/error) ** 2 - np.log(error) - 0.5 * np.log(2 * np.pi)
