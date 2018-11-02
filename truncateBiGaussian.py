#!/usr/bin/python3

import tpg
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Generate/Load existing two unconditional gaussian realizations using different variogram models

gaussian1 = tpg.genGaussian2DSim_FFT(120, 120, 0.5, 0.5, 'exponential', 30, 9)
model = tpg.model('spherical', 1, 10, 0.5, 45)
gaussian2 = tpg.genGaussian2DSim_SGSim(120, 120, 0.5, 0.5, model, 0)

#gaussian1 = np.loadtxt('gaussian1.txt') # if files exist
#gaussian2 = np.loadtxt('gaussian2.txt')

# Display each gaussian realization
plt.close()
fig, ax = plt.subplots()
im = plt.imshow(gaussian1, origin='lower', cmap='rainbow')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=5)
plt.savefig('gaussian1.png', dpi=300)

fig, ax = plt.subplots()
im = plt.imshow(gaussian2, origin='lower', cmap='rainbow')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=5)
plt.savefig('gaussian2.png', dpi=300)

# Truncate according to given rock type rule 
#thresholdLines = tpg.randTruncLines() # generate two random threshold lines 
#faciesMap = tpg.truncBiGaussian24facies(gaussian1, gaussian2, thresholdLines) # generate facies map with thresholds given as lines
faciesMap = tpg.truncBiGaussian23facies(gaussian1, gaussian2, 2, [-0.1, 0]) # generate facies map with thresholds given as list of values
np.savetxt('faciesMap.txt', faciesMap, fmt='%d')

# Display the derived facies map with a discrete colorbar 
tpg.discrete_imshow(faciesMap)
plt.savefig('faciesMap.png', dpi=300)

plt.show()
