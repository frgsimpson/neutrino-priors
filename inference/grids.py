""" Define a data object which holds our likelihood and posterior distribution on a grid. """
from dataclasses import dataclass

import numpy as np

N_DEFAULT_SAMPLES = 100_000


@dataclass
class LikelihoodGrid:
    """ A grid of values in the sigma-mu plane holding the prior, likelihood and posterior.
    Prior is not necessarily normalised. """
    sigmaArray: np.array
    muArray: np.array
    loglikelihood: np.ndarray
    n_samples: int
    prior_power: int = 1

    @property
    def logprior(self):
        sigma_prior = -self.prior_power * self.sigmaArray
        n_mu = len(self.muArray)
        return np.tile(sigma_prior, (n_mu, 1))

    @property
    def logposterior(self):
        return self.loglikelihood + self.logprior


def get_default_grid(n_sigma: int = 100, n_mu: int = 101, n_samples: int = N_DEFAULT_SAMPLES, log_mass_spacing: bool = True):

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