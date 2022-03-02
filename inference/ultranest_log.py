""" Estimate the model evidence using nested sampling. """

import numpy as np
import ultranest as ultra
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


def run_ultranest(hierarchy: Hierarchy, data: NeutrinoConstraint):

    run_name = get_savefile_name(hierarchy, data)
    dir_name = "ultra_" + run_name

    sampler = ultra.NestedSampler(PARAM_NAMES, evaluate_log_likelihood_of_parameters, prior_map, num_live_points=400,
        log_dir=dir_name, resume='resume')
    sampler.run()

