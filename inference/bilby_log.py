""" Estimate the model evidence using nested sampling. """

import numpy as np
import bilby
from scipy.stats import norm
from inference.sampling import log_pdf
from inference.utils import get_savefile_name
from neutrinos.constraints import load_neutrino_constraints, NeutrinoConstraint
from neutrinos.hierarchies import Hierarchy


MAX_LOG_MU = 0.  # Upper bound of 1 eV
MIN_LOG_MU = -7
MAX_LOG_SIGMA = 2
MIN_LOG_SIGMA = -5
PARAM_NAMES = ['mu', 'sigma', 'mL', 'mM', 'mH']

HIERARCHY = Hierarchy.Normal
DATA = load_neutrino_constraints()


def prior_map(cube):
    """ Our two parameters are log mu and log sigma """

    mu_value = (MAX_LOG_MU - MIN_LOG_MU) * cube[0] + MIN_LOG_MU
    sigma = np.exp((MAX_LOG_SIGMA - MIN_LOG_SIGMA) * cube[1] + MIN_LOG_SIGMA)
    log_masses = norm.ppf(cube[2:], loc=mu_value, scale=sigma)
    masses = np.exp(log_masses)
    m1 = masses[0]
    m2 = masses[1]
    m3 = masses[2]

    return [mu_value, sigma, m1, m2, m3]


class HierarchyLikelihood(bilby.Likelihood):
    def __init__(self, data):
        """
        A Gaussian likelihood

        Parameters
        ----------
        data: array_like
            The data to analyse
        """
        super().__init__(parameters={'logmu': None, 'sigma': None, 'm1q': None, 'm2q': None, 'm3q': None})
        self.data = data
        self.N = 3 # len(data)

    def log_likelihood(self):
        quantile_vector = np.array([self.parameters['m1q'], self.parameters['m2q'], self.parameters['m3q']])
        log_masses = norm.ppf(quantile_vector, loc=self.parameters['logmu'], scale=self.parameters['sigma'])
        mass_vector = np.exp(log_masses)

        if not np.all(np.diff(mass_vector) >= 0):
            return -1e100  # Faster to exclude unsorted region then compensate Z for missing volume. (factor of 6)

        if HIERARCHY == Hierarchy.Normal:
            msqr1 = mass_vector[1] ** 2 - mass_vector[0] ** 2  # M ^ 2 - S ^ 2 ie m2 - m1
            msqr2 = mass_vector[2] ** 2 - mass_vector[1] ** 2  # L ^ 2 - M ^ 2 ie m3 - m2
        else:
            msqr1 = mass_vector[2] ** 2 - mass_vector[1] ** 2  # L ^ 2 - S ^ 2 % m2 - m3
            msqr2 = mass_vector[2] ** 2 - mass_vector[0] ** 2  # L ^ 2 - M ^ 2 % m2 - m1

        mass_sum = np.sum(mass_vector)

        loglikeli = log_pdf(msqr1, self.data.m21_sqr, self.data.m21_sqr_error)   # Smaller mass gap
        loglikeli += log_pdf(msqr2, self.data.m31_sqr, self.data.m31_sqr_error)  # Larger mass gap
        loglikeli += log_pdf(mass_sum, self.data.sum_of_masses_offset, self.data.sum_of_masses_one_sigma)

        return loglikeli


def evaluate_log_likelihood_of_parameters(param_vector):

    param_vector = param_vector[2:]  # Remove hyperparameters
    # Order the neutrino masses for ease of likelihood eval
    if not np.all(np.diff(param_vector) >= 0):
        return -1e100  # Faster to exclude unsorted region then compensate Z for missing volume. (factor of 6)

    if HIERARCHY == Hierarchy.Normal:
        msqr1 = param_vector[1] ** 2 - param_vector[0] ** 2  # M ^ 2 - S ^ 2 ie m2 - m1
        msqr2 = param_vector[2] ** 2 - param_vector[1] ** 2  # L ^ 2 - M ^ 2 ie m3 - m2
    else:
        msqr1 = param_vector[2] ** 2 - param_vector[1] ** 2  # L ^ 2 - S ^ 2 % m2 - m3
        msqr2 = param_vector[2] ** 2 - param_vector[0] ** 2  # L ^ 2 - M ^ 2 % m2 - m1

    mass_sum = np.sum(param_vector)

    loglikeli = log_pdf(msqr1, DATA.m21_sqr, DATA.m21_sqr_error)   # Smaller mass gap
    loglikeli += log_pdf(msqr2, DATA.m31_sqr, DATA.m31_sqr_error)  # Larger mass gap
    loglikeli += log_pdf(mass_sum, DATA.sum_of_masses_offset, DATA.sum_of_masses_one_sigma)

    return loglikeli


def get_bilby_priors():

    priors = {'logmu': bilby.prior.Uniform(minimum=np.log(5e-4), maximum=np.log(0.3), name='logmu', latex_label='log \mu'),
              'sigma': bilby.prior.LogUniform(minimum=5e-3, maximum=20, name='sigma', latex_label='\sigma'),
              'm1q': bilby.prior.Uniform(minimum=0, maximum=1, name='m1q', latex_label='m1q'),
              'm2q': bilby.prior.Uniform(minimum=0, maximum=1, name='m2q', latex_label='m2q'),
              'm3q': bilby.prior.Uniform(minimum=0, maximum=1, name='m3q', latex_label='m3q')}
    return priors


def run_bilby(hierarchy: Hierarchy, data: NeutrinoConstraint, nlive: int = 1000):

    hierarchy_str = 'nh' if hierarchy == Hierarchy.Normal else 'ih'
    sum_str = str(data.sum_of_masses_one_sigma)[:5]  # Max 5 sig fig
    prefix = '' if data.sum_of_masses_offset == 0.else str(data.sum_of_masses_offset) + '_'
    sample = "rwalk"
    label = sample + str(nlive) + "_" + prefix + 'likeli_' + hierarchy_str + '_' + sum_str

    #  Going to get posteriors on the quantile, then need to do some post-processing
    likelihood = HierarchyLikelihood(data)
    priors = get_bilby_priors()
    bilby.run_sampler(likelihood, priors, sampler="dynesty", nlive=nlive, label=label, bound="multi", sample=sample, dlogz=0.01)



