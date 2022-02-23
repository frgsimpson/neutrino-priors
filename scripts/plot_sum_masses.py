from matplotlib import pyplot as plt
import numpy as np

from inference.calc_likelihood import get_posterior
from neutrinos.constraints import load_neutrino_constraints
from neutrinos.hierarchies import Hierarchy


N_MASS_BINS = 60
SUM_OF_MASSES_ONE_SIGMA = np.array([0.089, 0.099, 0.12]) * 0.5
SUM_OF_MASSES_OFFSET = [0, 0, 0]  # Corresponding offsets for each upper bound
LINESTYLES = ['-', '--', ':']
COLOURS = ['b', 'r', 'k']


def make_sum_of_masses_plot():
    """ Plot the poserior distribution for the sum of neutrino masses for two hierarchies and different upper bounds. """

    data = load_neutrino_constraints()
    plt.figure(figsize=(12, 6))

    for i, hierarchy in enumerate(Hierarchy):
        for j, upper_bound in enumerate(SUM_OF_MASSES_ONE_SIGMA):
            data.sum_of_masses_one_sigma = upper_bound
            data.sum_of_masses_offset = SUM_OF_MASSES_OFFSET[j]
            linestyle = LINESTYLES[j]

            likeligrid = get_posterior(hierarchy, data)

            y = likeligrid.mass_posterior[:, 3]
            plt.plot(likeligrid.mass_log_bins, y, linestyle=linestyle, color=COLOURS[j])

            plt.xlabel('m/eV')
            plt.ylabel('p(m)')

    print_figure()
    plt.show()


def print_figure():

    filename = './plots/sum_of_masses_posteriors.png'
    plt.savefig(filename,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0)
    print('Saved to file:', filename)


make_sum_of_masses_plot()
