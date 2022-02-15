from typing import List

from matplotlib import pyplot as plt
import numpy as np
from scipy.special import logsumexp
from skimage.filters import gaussian as gaussian_filter

from inference.calc_likelihood import get_posterior
from neutrinos.constraints import load_neutrino_constraints
from inference.grids import LikelihoodGrid
from neutrinos.hierarchies import Hierarchy

KERNEL_WIDTH = 5.  # How many sigma to smooth across
SMOOTH_SIGMA = 2   # How wide to smooth
COLOR_RANGE = 10   # How far does the log likelihood fall before it becomes black?
COLOR_BUFFER = 0.  # how much of a gap between white and the colour of the max value


def make_mu_sigma_figures(sum_of_masses_one_sigma=0.0445):
    """ Make plots for the two hierarchies with matching colour scheme. """

    data = load_neutrino_constraints(sum_of_masses_one_sigma)

    posterior_list = []
    for i, hierarchy in enumerate(Hierarchy):
        posterior = get_posterior(hierarchy, data)
        posterior_list.append(posterior)

    print_evidence(posterior_list)
    make_plots(posterior_list, data.sum_of_masses_one_sigma)


def print_evidence(grids: List[LikelihoodGrid]):
    """ Take NH and IH results and compute the log z values. """

    nh_log_z = calculate_log_evidence(grids[0])
    ih_log_z = calculate_log_evidence(grids[1])
    print('Evidence:', nh_log_z)
    print('Evidence:', ih_log_z)
    print('Ratio:', np.exp(nh_log_z - ih_log_z))


def make_plots(grids: List[LikelihoodGrid], sum_of_masses_one_sigma):
    """ Make two subplots from NH and IH in mu-sigma plane. """
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False)

    smooth_likelihoods = []
    for grid in grids:
        smoothed = apply_smoothing(grid.logposterior, SMOOTH_SIGMA)
        smooth_likelihoods.append(smoothed)

    maxval = np.maximum(np.max(smooth_likelihoods[0]), np.max(smooth_likelihoods[1]))
    minval = maxval - COLOR_RANGE
    vmax = maxval + COLOR_BUFFER

    for (ax, grid, log_likeli) in zip(axes.flat, grids, smooth_likelihoods):

        extent = [np.log(grid.muArray[0]), np.log(grid.muArray[-1]), np.log(grid.sigmaArray[0]), np.log(grid.sigmaArray[-1])]
        im = ax.imshow(log_likeli.T,  cmap='hot', vmin=minval, vmax=vmax, origin='lower',  extent=extent, aspect=0.6)


        # todo Fix the ticks
        # x_indices =
        # ax.set_xticks([20, 40, 60, 80])
        # ax.set_xticklabels(x_label_list)
        # plt.xticks(grid.muArray)
        # plt.yticks(np.log(grid.sigmaArray))

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)

    # Adding an outer axes allows for a shared xlabel
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('log ($\mu$ / eV)', labelpad=-60)
    plt.ylabel('log $\sigma$')
    plt.grid(False)

    print_figure(sum_of_masses_one_sigma)
    plt.show()


def calculate_log_evidence(grid: LikelihoodGrid):
    """ Evidence is the expectation of the likelihood under the prior. """
    return logsumexp(grid.logposterior)


def print_figure(mass_bound):

    filename = './plots/mu_sigma_' + str(mass_bound) + '.png'
    plt.savefig(filename,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0)
    print('Saved to file:', filename)


def apply_smoothing(image, smooth_sigma):
    return gaussian_filter(image, sigma=smooth_sigma, mode='nearest', truncate=KERNEL_WIDTH)


if __name__ == '__main__':
    make_mu_sigma_figures()


