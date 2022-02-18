""" Estimate the model evidence using nested sampling. """

import dynesty as dy
import numpy as np

from inference.sampling import log_pdf
from neutrinos.constraints import load_neutrino_constraints
from neutrinos.hierarchies import Hierarchy

MAX_MASS = 1.0
DATA = load_neutrino_constraints()
HIERARCHY = Hierarchy.Normal


def prior_map(cube):
    """ Our three masses have uniform priors so mapping of the prior quantile is trivial. """

    return cube * MAX_MASS


def evaluate_log_likelihood_of_parameters(param_vector):

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


# Move to separate script
sampler = dy.NestedSampler(loglikelihood=evaluate_log_likelihood_of_parameters,
                           prior_transform=prior_map,
                           ndim=3,
                           nlive=500,
                           bound='multi',
                           sample='auto')

sampler.run_nested(dlogz=0.01, maxiter=100_000)
sampler.results.summary()

# iter: 11549 | +500 | bound: 77 | nc: 1 | ncall: 60584 | eff(%): 19.888 | loglstar:   -inf < 22.636 <    inf | logz:  4.172 +/-    nan | dlogz:  0.000 >  0.010
# nlive: 500
# niter: 11840
# ncall: 49225
# eff(%): 25.069
# logz:  3.589
