""" Define a data object which holds measurements of the mass splittings and the sum of the masses. """
from dataclasses import dataclass


@dataclass
class NeutrinoConstraint:
    """ A collection of constraints on the three neutrino masses."""

    # New constraints
    m21_sqr: float
    m21_sqr_error: float
    m31_sqr: float
    m31_sqr_error: float
    sum_of_masses_offset: float = 0.
    sum_of_masses_one_sigma: float = 0.1 * 0.5


def load_default_neutrino_constraints(sum_of_masses_one_sigma: float = 0.06) -> NeutrinoConstraint:

    return NeutrinoConstraint(
        m21_sqr=7.49e-5,
        m31_sqr=2.484e-3,
        m21_sqr_error=0.18e-5,
        m31_sqr_error=0.045e-3,
        sum_of_masses_one_sigma=sum_of_masses_one_sigma,
    )



