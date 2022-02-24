from matplotlib import pyplot as plt
import numpy as np

from inference.calc_likelihood import get_posterior
from inference.utils import print_figure
from neutrinos.constraints import load_neutrino_constraints
from neutrinos.hierarchies import Hierarchy

SUM_OF_MASSES_ONE_SIGMA = np.array([0.089, 0.099, 0.12]) * 0.5
SUM_OF_MASSES_OFFSET = [0, 0, 0]  # Corresponding offsets for each upper bound
LINESTYLES = ['-', '--', ':']
COLOURS = ['b', 'r', 'k']
plt.style.use("../jcappaper.mplstyle")


def make_sum_of_masses_plot(log_mass: bool = False):
    """ Plot the posterior distribution for the sum of neutrino masses for two hierarchies and
     a range of upper bounds."""

    data = load_neutrino_constraints()

    for i, hierarchy in enumerate(Hierarchy):
        for j, upper_bound in enumerate(SUM_OF_MASSES_ONE_SIGMA):
            data.sum_of_masses_one_sigma = upper_bound
            data.sum_of_masses_offset = SUM_OF_MASSES_OFFSET[j]
            linestyle = LINESTYLES[j]

            likeligrid = get_posterior(hierarchy, data)

            y = likeligrid.mass_posterior[:, 3]
            if log_mass:
                plt.plot(likeligrid.mass_log_bins, y, linestyle=linestyle, color=COLOURS[j])
            else:
                mass = np.exp(likeligrid.mass_log_bins)
                y /= mass  # p(m) = p(log m) / m
                y /= np.sum(y)
                plt.plot(mass, y, linestyle=linestyle, color=COLOURS[j])

            plt.xlabel('$\Sigma_\\nu$ [eV]')
            plt.ylabel('p($\Sigma_\\nu$ )')
            plt.xlim([0, 0.3])

    print_figure('sum_of_masses_posteriors')
    plt.show()

make_sum_of_masses_plot()
