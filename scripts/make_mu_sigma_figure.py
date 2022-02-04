from typing import List

from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import gaussian as gaussian_filter

from inference.calc_likelihood import get_posterior
from neutrinos.constraints import load_default_neutrino_constraints
from inference.grids import LikelihoodGrid
from neutrinos.hierarchies import Hierarchy

KERNEL_WIDTH = 5.  # How many sigma to smooth across
SMOOTH_SIGMA = 2   # How wide to smooth
COLOR_RANGE = 10   # How far does the log likelihood fall before it becomes black?
COLOR_BUFFER = 0.  # how much of a gap between white and the colour of the max value


def make_mu_sigma_figures(sum_of_masses_one_sigma=0.04):
    """ Make plots for the two hierarchies with matching colour scheme. """

    data = load_default_neutrino_constraints(sum_of_masses_one_sigma)

    posterior_list = []
    for i, hierarchy in enumerate(Hierarchy):
        posterior = get_posterior(hierarchy, data)
        posterior_list.append(posterior)

    make_plots(posterior_list, data.sum_of_masses_one_sigma)


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
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('log ($\mu$ / eV)', labelpad=-60)
    plt.ylabel('log $\sigma$')
    plt.grid(False)

    print_figure(sum_of_masses_one_sigma)
    plt.show()


def print_figure(mass_bound):

    filename = './plots/mu_sigma_' + str(mass_bound) + '.png'
    plt.savefig(filename,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0)
    print('Saved to file:', filename)


def apply_smoothing(image, smooth_sigma):
    return gaussian_filter(image, sigma=smooth_sigma, mode='nearest', truncate=KERNEL_WIDTH)


# smooth_likelihood = grid.posterior
# smooth_likelihood = gaussian_filter(grid.posterior, sigma=smooth_sigma, mode='nearest', truncate=2.0)
# log_smooth_likelihood -= np.max(log_smooth_likelihood) -1 # Normalise for cleaner numerics
# likelihood = np.exp(log_smooth_likelihood)
# log_smooth_likelihood = np.log(1 + smooth_likelihood)
# x_label_list = grid.muArray
# ax.set_xticks(np.linspace(-1, 1, len(x_label_list)))
# ax.set_xticklabels(x_label_list)
# maxval = np.max(log_smooth_likelihood)
#
# mesh = ax.pcolormesh(log_smooth_likelihood, cmap=cm)
# mesh.set_clim(vmin, vmax)


if __name__ == '__main__':
    make_mu_sigma_figures()

# def setup_colorbar():
#     EPSILON = 1e-10;
#     FONTCOLOR = [1 - EPSILON, 1, 1];
#     FONTSIZE = 11;
#
#     N_TICKS = 6;
#
#     cb = colorbar('south')
#     cax = caxis
#     colormap('hot')
#
#     N_SIG_FIG = 2;
#     zMAX = cax(2);
#     zTicks = linspace(0, zMAX, N_TICKS); % [0 1 2 3 4 5 zMAX];
#     pMAX = exp(zMAX) - 1;
#     pTicks = exp(zTicks) - 1;
#
#     pTicksNorm = round(pTicks. / pMAX, N_SIG_FIG)
#     pTicksNorm(2) = round(pTicks(2). / pMAX, N_SIG_FIG + 1)
#
#     LIMITS = [0 zMAX]

    # set(cb, 'FontSize', FONTSIZE, 'YTick', zTicks, 'YTickLabel', pTicksNorm, 'Color', FONTCOLOR, 'Limits', LIMITS);

    # XPOS = 0.15; % 0.88
    # YPOS = 0.20;
    #
    #
    # WIDTH = 0.6;
    # HEIGHT = 0.02;
    #
    # set(cb, 'position', [XPOS, YPOS, WIDTH, HEIGHT])



#
# bayesFactor = NHoutput.evidence / IHoutput.evidence;
# peakFactor = NHoutput.peak / IHoutput.peak
#
# nh_evidence = NHoutput.evidence
# ih_evidence = IHoutput.evidence
#
# function
# initialise_constants()
#
# DO_COLORBAR = true;
#
# NH_INDEX = 6;
# IH_INDEX = 7;
#
# if (~exist('SUM_ERROR', 'var'))
#     SUM_ERROR = 10 / 2;
# end
#
# if (~exist('SIGMA_OFFSET', 'var'))
#     SIGMA_OFFSET = 0;
# end
#
# if (~exist('INV_PRIOR', 'var'))
#     INV_PRIOR = false;
# end
#
# if (~exist('INV_PRIOR', 'var'))
#     INV_PRIOR = false;
# end
#
# if (~exist('DO_EVIDENCE_PLOT', 'var'))
#     DO_EVIDENCE_PLOT = false
# end
#
# if (~exist('DO_PRINT', 'var'))
#     DO_PRINT = true
# end
#
# if (INV_PRIOR)
#     NH_FILENAME = './plots/invNH_map';
#     IH_FILENAME = './plots/invIH_map';
#     EVIDENCE_FILENAME = './plots/invevidence';
# else
#     NH_FILENAME = './plots/NH_map';
#     IH_FILENAME = './plots/IH_map';
#     EVIDENCE_FILENAME = './plots/evidence';
# end
#
# end
# end
#
# function
# plot_evidence_map(NHoutput, logEvidenceMap)
#
# FONT_SIZE = 21;
#
# MAX_EVIDENCE = 10;
# logEvidenceMap(logEvidenceMap > MAX_EVIDENCE) = MAX_EVIDENCE;
#
# logEvidenceMap(isnan(logEvidenceMap)) = MAX_EVIDENCE;
#
# % Tiny
# bit
# more
# smoothing
# where
# we
# hit
# maxima
# SIGMA_SMOOTH = 7; % width
# of
# kernel
# logEvidenceMap = imgaussfilt(logEvidenceMap, SIGMA_SMOOTH);
#
# X = NHoutput.likeli.muArray;
# Y = log(NHoutput.likeli.sigmaArray);
#
# surf(X, Y, logEvidenceMap
# ','
# EdgeColor
# ','
# None
# '); \
#                                             % set(gca,'
# yscale
# ','
# log
# ');
# set(gca, 'xscale', 'log');
# shading
# interp
#
# YMAX = 20; % actual
# max is 40
# but
# need
# last
# 10
# for smoothing
# YMIN = 0.1;
#
# ylim([log(YMIN) log(YMAX)])
#
# xlim([min(X) max(X)])
#
# view(2);
#
# CMIN = 1;
# CMAX = MAX_EVIDENCE;
# caxis([CMIN CMAX])
#
# DO_EVIDENCE = true;
# setup_colorbar(DO_EVIDENCE)
#
# tickVals =[0.001     0.01     0.1   1];
# % muvals = log(tickVals);
# set(gca, 'XTick', tickVals); % xticks(muvals);
# XTickLabels = {'0.001';   '0.01'; '0.1'; '1'};
#
# YtickVals =[-2 -1 0 1 2];
# % muvals = log(tickVals);
# set(gca, 'YTick', YtickVals);
#
# set(gca, 'XTickLabel', XTickLabels, 'FontSize', 16);
# xlabel('\mu (eV)', 'FontSize', FONT_SIZE)
# % ylabel('\sigma', 'FontSize', FONT_SIZE)
#
# ylabel('log \sigma', 'FontSize', FONT_SIZE, 'Rotation', 90);
#
# % set(gca, 'FontSize', 16);
#
# end
#
# function[C, cax] = get_colormap(DO_COLORBAR)
# INVERT_COLORMAP = false;
#
# % NHoutput.h,
# if (DO_COLORBAR)
# setup_colorbar();
# end
#
# if (INVERT_COLORMAP)
# colormap(flipud(colormap)); % INVERT
# end
# C=colormap;
# cax = caxis;
# box on
#
# end
#
# function setup_colorbar(DO_EVIDENCE_RATIO)
#
# EPSILON = 1e-10;
# FONTCOLOR =[1-EPSILON, 1, 1];
# FONTSIZE =  11;
#
# N_TICKS = 6;
#
# cb = colorbar('south');
# cax = caxis;
# colormap('hot'); % 'bone', jet, default, hot
#
# if (~exist('DO_EVIDENCE_RATIO', 'var'))
# DO_EVIDENCE_RATIO = false;
# end
#
# if (DO_EVIDENCE_RATIO)
# N_SIG_FIG = -1;
# zMAX = cax(2); zMIN = 1;
# zTicks = linspace(zMIN, zMAX, N_TICKS); %[0 1 2 3 4 5 zMAX];
# pMAX = 1;
# pTicks = round(exp(zTicks));
# else
# N_SIG_FIG = 2;
# zMAX = cax(2);
# zTicks = linspace(0, zMAX, N_TICKS); %[0 1 2 3 4 5 zMAX];
# pMAX = exp(zMAX) - 1;
# pTicks = exp(zTicks) - 1;
# end
#
# pTicksNorm = round(pTicks./ pMAX, N_SIG_FIG);
# pTicksNorm(2) = round(pTicks(2)./ pMAX, N_SIG_FIG + 1);
#
# LIMITS =[0 zMAX];
#
# set(cb, 'FontSize', FONTSIZE, 'YTick', zTicks, 'YTickLabel', pTicksNorm, 'Color', FONTCOLOR, 'Limits', LIMITS);
#
# % Now set position
# if (DO_EVIDENCE_RATIO)
# XPOS = 0.15;
# YPOS = 0.85;
#
# else
# XPOS = 0.15; % 0.88
# YPOS = 0.20;
# end
#
# WIDTH = 0.6;
# HEIGHT = 0.02;
#
# set(cb, 'position', [XPOS, YPOS, WIDTH, HEIGHT]);
#
# end
#
# function set_colormap(C, cax, DO_COLORBAR)
#
# set(gcf, 'ColorMap', C);
# caxis(cax);
# box on
#
# if (DO_COLORBAR)
# setup_colorbar();
# end
#
#
# end
#
# function write_text(DO_NH)
#
# ANGLE = 0;
#
# if (DO_NH)
# H_String = 'NORMAL HIERARCHY';
# else
# H_String = 'INVERTED HIERARCHY';
# end
# alignment = 'left';
# FontSize = 16;
# COLOR = 'w';
#
# xLoc = 0.065;
# yLoc = 0.17;
#
# h = add_text_to_plot(H_String, xLoc, yLoc, FontSize, alignment, COLOR);
#
# set(h, 'Rotation', ANGLE);
#
# end
#
# function write_evidence_text()
# % weirdly doesnt work
# ANGLE = 0;
#
# H_String = 'BAYES FACTOR';
# alignment = 'left';
# FontSize = 16;
# COLOR = 'w';
#
# xLoc = 0.065;
# yLoc = 1 - 0.17;
#
# t=uicontrol(...,
# 'Style', 'text', ...
# 'Units', 'norm', ...
# 'Position', [0.18 0.6 0.10 0.20], ...
# 'BackgroundColor', 'none', ...
# 'ForegroundColor', 'w', ...
# 'String', H_String);
# % h = add_text_to_plot(H_String, xLoc, yLoc, FontSize, alignment, COLOR);
#
# set(h, 'Rotation', ANGLE);
#
# end
