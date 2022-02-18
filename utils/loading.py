""" Routines for saving and loading the computationally-intensive likelihood grids. """
import pickle

from neutrinos.constraints import NeutrinoConstraint
from inference.grids import LikelihoodGrid
from neutrinos.hierarchies import Hierarchy


def load_posterior(hierarchy, data, n_samples: int) -> LikelihoodGrid:

    filename = get_savefile_name(hierarchy, data, n_samples)

    with open(filename, 'rb') as f:
        posterior = pickle.load(f)

    return posterior


def save_posterior(hierarchy, data, posterior):

    filename = get_savefile_name(hierarchy, data, posterior.n_samples)

    with open(filename, 'wb') as f:
        pickle.dump(posterior, f)


def get_savefile_name(hierarchy: Hierarchy, data: NeutrinoConstraint, n_samples: int):
    """ Store data in file depending on the hierarchy, sum constraint, """

    hierarchy_str = 'nh' if hierarchy == Hierarchy.Normal else 'ih'
    sum_str = str(data.sum_of_masses_one_sigma)[:5]  # Max 5 sig fig
    sample_str = str(n_samples)
    prefix = '' if data.sum_of_masses_offset == 0.else str(data.sum_of_masses_offset) + '_'

    return prefix + 'likeli_' + hierarchy_str + '_' + sum_str + '_' + sample_str
