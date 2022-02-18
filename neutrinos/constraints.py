""" Define a data object which holds measurements of the mass splittings and the sum of the masses. """
from dataclasses import dataclass
from typing import Optional


@dataclass
class NeutrinoConstraint:
    """ A collection of constraints on the three neutrino masses."""

    m21_sqr: float
    m21_sqr_error: float
    m31_sqr: float
    m31_sqr_error: float
    sum_of_masses_one_sigma: float
    sum_of_masses_offset: float


def load_neutrino_constraints(sum_of_masses_one_sigma: Optional[float] = None,
                              sum_of_masses_offset: Optional[float] = 0.,
                              new_splittings: bool = True) -> NeutrinoConstraint:
    """ Constraints from particle physics and cosmology. """

    if sum_of_masses_one_sigma is None:
        # Use EBOSS Table 8 of https://arxiv.org/pdf/2007.08991.pdf
        sum_of_masses_offset = -0.026
        sum_of_masses_one_sigma = 0.060

    if new_splittings:
        constraints = NeutrinoConstraint(
            m21_sqr=7.42e-5,
            m31_sqr=2.5e-3,
            m21_sqr_error=0.21e-5,
            m31_sqr_error=0.027e-3,
            sum_of_masses_offset=sum_of_masses_offset,
            sum_of_masses_one_sigma=sum_of_masses_one_sigma,
        )
    else:
        constraints = NeutrinoConstraint(
            m21_sqr=7.49e-5,
            m31_sqr=2.484e-3,
            m21_sqr_error=0.18e-5,
            m31_sqr_error=0.045e-3,
            sum_of_masses_offset=sum_of_masses_offset,
            sum_of_masses_one_sigma=sum_of_masses_one_sigma,
        )

    return constraints


def load_pure_sum_masses(sum_of_masses_one_sigma: float = 0.060, sum_of_masses_offset: float = -0.026):
    return NeutrinoConstraint(
        m21_sqr=7.42e-5,
        m31_sqr=2.5e-3,
        m21_sqr_error=1e10,
        m31_sqr_error=1e10,
        sum_of_masses_offset=sum_of_masses_offset,
        sum_of_masses_one_sigma=sum_of_masses_one_sigma,
    )


def load_pure_splittings():
    return NeutrinoConstraint(
            m21_sqr=7.42e-5,
            m31_sqr=2.5e-3,
            m21_sqr_error=0.21e-5,
            m31_sqr_error=0.027e-3,
            sum_of_masses_offset=0.,
            sum_of_masses_one_sigma=1e10,
        )
