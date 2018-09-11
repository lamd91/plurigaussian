#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
import tpg
import variomodels

# Simulate one continuous gaussian realizations
gaussian = tpg.genGaussianSim_2D(30, 30, 1, 1, 'exponential', 10)

# Display and save each generated gaussian realizations
plt.imshow(gaussian, cmap='rainbow', aspect='auto')
plt.colorbar()
plt.savefig('gaussian0.png', dpi=300)

# Truncate first generated gaussian realization into 3 facies according to facies proportions
faciesMap = tpg.truncGaussian2facies(gaussian, 0.3, 0.2, 0.5)

# Display the derived facies map with a discrete colorbar
plt.close()
tpg.discrete_imshow(faciesMap)
plt.savefig('faciesMap0.png', dpi=300)
