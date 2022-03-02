from matplotlib import pyplot as plt
import numpy as np

from inference.calc_likelihood import get_posterior
from inference.utils import print_figure
from neutrinos.constraints import load_neutrino_constraints
from neutrinos.hierarchies import Hierarchy

N_SAMPLES = 10_000
SUM_OF_MASSES_ONE_SIGMA = np.array([0.12, 0.102, 0.89]) * 0.5
SUM_OF_MASSES_OFFSET = [0, 0, 0]  # Corresponding offsets
LINESTYLES = ['-', '--', ':']
COLOURS = ['b', 'r', 'k']
plt.style.use("../jcappaper.mplstyle")


def make_mass_plot(log_mass: bool = False):

    data = load_neutrino_constraints()
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(12, 6))

    for i, hierarchy in enumerate(Hierarchy):
        for j, upper_bound in enumerate(SUM_OF_MASSES_ONE_SIGMA):
            data.sum_of_masses_one_sigma = upper_bound
            data.sum_of_masses_offset = SUM_OF_MASSES_OFFSET[j]
            linestyle = LINESTYLES[j]

            likeligrid = get_posterior(hierarchy, data, n_samples=N_SAMPLES)

            for m in range(3):
                y = likeligrid.mass_posterior[:, m]

                if log_mass:
                    axes[i].plot(likeligrid.mass_log_bins, y, linestyle=linestyle, color=COLOURS[m])
                else:
                    # mass = np.exp(likeligrid.mass_log_bins)
                    # y /= mass  # p(m) = p(log m) / m
                    mass = likeligrid.mass_bins
                    y /= np.sum(y)
                    axes[i].plot(mass, y, linestyle=linestyle, color=COLOURS[m])

        axes[i].set_xlabel('m/eV')
        axes[i].set_ylabel('p(m)')
        axes[i].set_xlim([0, 0.09])

    print_figure('mass_posteriors')
    plt.show()

make_mass_plot()
