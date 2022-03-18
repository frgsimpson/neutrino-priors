""" Define a data object which holds our likelihood and posterior distribution on a grid. """
from dataclasses import dataclass

import numpy as np

N_DEFAULT_SAMPLES = 10_000
N_MASS_BINS = 200  # For binning the posterior - 200 bins from 0-0.2 eV
MAX_MASS_PLOT = 0.2
N_PRIOR_BINS = 256  # For binning mu and sigma prior grid

@dataclass
class LikelihoodGrid:
    """ A grid of values in the sigma-mu plane holding the prior, likelihood and posterior.
    Prior is not necessarily normalised. """
    sigmaArray: np.array
    muArray: np.array
    loglikelihood: np.ndarray
    n_samples: int
    prior_power: int = 0
    mass_posterior = np.zeros((N_MASS_BINS, 4))  # Final column is for the sum of masses
    mass_log_bins = np.linspace(-7, 0, N_MASS_BINS)
    mass_bins = np.linspace(0., MAX_MASS_PLOT, N_MASS_BINS)  # For plotting

    @property
    def logprior(self):
        """ Add a prior of the form sigma^-(1+prior_power).
         1 comes from the 1/sigma prior due to the log distribution of sigmaArray """
        sigma_prior = -self.prior_power * self.sigmaArray
        n_mu = len(self.muArray)
        return np.tile(sigma_prior, (n_mu, 1))

    @property
    def logposterior(self):
        return self.loglikelihood + self.logprior


def get_default_grid(n_sigma: int = N_PRIOR_BINS, n_mu: int = N_PRIOR_BINS, n_samples: int = N_DEFAULT_SAMPLES, log_mass_spacing: bool = True):

    muMIN = 5e-4
    muMAX = 0.3

    sigmaMIN = 5e-3
    sigmaMAX = 20

    sigmaArray = np.logspace(np.log10(sigmaMIN), np.log10(sigmaMAX), n_sigma)
    if log_mass_spacing:
        muArray = np.logspace(np.log10(muMIN), np.log10(muMAX), n_mu)
    else:
        muArray = np.linspace(muMIN, muMAX, n_mu)
    zero_likelihood = np.zeros((n_mu, n_sigma))

    return LikelihoodGrid(sigmaArray=sigmaArray, muArray=muArray, loglikelihood=zero_likelihood, n_samples=n_samples)
