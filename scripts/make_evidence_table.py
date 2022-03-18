""" Print a collection of evidence ratios """
import numpy as np
from typing import List

from inference.calc_likelihood import get_posterior
from inference.grids import LikelihoodGrid
from neutrinos.constraints import load_neutrino_constraints
from neutrinos.hierarchies import Hierarchy
from scripts.make_mu_sigma_figure import print_evidence, calculate_log_evidence

TWO_SIGMA_LIMITS = [0.8, 0.12, 0.102, 0.099, 0.089, 0.099]
OFFSETS = [0, 0, 0, 0, 0, 0.04]
DELTA_CHI2_BOOSTS = [3.669, 33]


def print_evidence(grids: List[LikelihoodGrid], delta_chi2_boost: float):
    """ Take NH and IH results and compute the log z values. """

    nh_log_z = calculate_log_evidence(grids[0])
    ih_log_z = calculate_log_evidence(grids[1])
    K = np.exp(nh_log_z - ih_log_z)
    K_CHI_SQR = K * delta_chi2_boost
    print('Ratio:', K)
    print('With chi sqr:', K_CHI_SQR)


def calculate_evidence_value(sum_of_masses_one_sigma: float, offset: float, new_splittings: bool):
    """ Get evidences for the two hierarchies. """

    data = load_neutrino_constraints(sum_of_masses_one_sigma, sum_of_masses_offset=offset, new_splittings=new_splittings)

    posterior_list = []
    for i, hierarchy in enumerate(Hierarchy):
        posterior = get_posterior(hierarchy, data)
        posterior_list.append(posterior)

    delta_chi2_boost = DELTA_CHI2_BOOSTS[1] if new_splittings else DELTA_CHI2_BOOSTS[0]

    print_evidence(posterior_list, delta_chi2_boost)


for new_split in [True, False]:
    for (offset, twosigma) in zip(OFFSETS, TWO_SIGMA_LIMITS):
        calculate_evidence_value(sum_of_masses_one_sigma=twosigma * 0.5, offset=offset, new_splittings=new_split)
