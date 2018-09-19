#!/usr/bin/python3

import tpg
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Simulate two continuous gaussian realizations using different variogram models
gaussian1 = tpg.genGaussianSim_2D_aniso(120, 120, 0.5, 0.5, 'exponential', 10, 0.6, 45)
gaussian2 = tpg.genGaussianSim_2D_aniso(120, 120, 0.5, 0.5, 'exponential', 20, 0.25, 0)

# Display each generated gaussian realizations
plt.close()
fig, ax = plt.subplots()
im = plt.imshow(gaussian1, origin='lower', cmap='rainbow')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=5)
np.savetxt('gaussian1.txt', gaussian1)
plt.savefig('gaussian1.png', dpi=300)

plt.close()
fig, ax = plt.subplots()
im = plt.imshow(gaussian2, origin='lower', cmap='rainbow')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=5)
np.savetxt('gaussian2.txt', gaussian2)
lt.savefig('gaussian2.png', dpi=300)
plt.close()

# Truncate according to given rock type rule 
thresholdLines = tpg.truncLines() # generate two random threshold lines 
faciesMap = tpg.truncBiGaussian2facies(gaussian1, gaussian2, thresholdLines)

# Display the derived facies map with a discrete colorbar 
plt.close()
tpg.discrete_imshow(faciesMap)
plt.savefig('faciesMap.png', dpi=300)

