import numpy as np

from inference.bilby_log import run_bilby
from neutrinos.constraints import load_neutrino_constraints
from neutrinos.hierarchies import Hierarchy

SUM_OF_MASSES_ONE_SIGMA = np.array([0.12, 0.102, 0.89]) * 0.5
SUM_OF_MASSES_OFFSET = [0, 0, 0]  # Corresponding offsets
data = load_neutrino_constraints()

for i, hierarchy in enumerate(Hierarchy):
    for j, upper_bound in enumerate(SUM_OF_MASSES_ONE_SIGMA):
        data.sum_of_masses_one_sigma = upper_bound
        data.sum_of_masses_offset = SUM_OF_MASSES_OFFSET[j]

        run_bilby(hierarchy, data)
