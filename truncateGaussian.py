#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myFunctions import makeFlowParFileForGW
import tpg

# Simulate one continuous gaussian realization

model = tpg.model('exponential', 1, 10, 0.5, 60)
gaussian = tpg.genGaussian2DSim_SGSim(120, 120, 0.5, 0.5, model, 0)

# Display generated gaussian realization
plt.close()
fig, ax = plt.subplots()
im = plt.imshow(gaussian, origin='lower', cmap='rainbow')
divider = make_axes_locatable(ax) 
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(im, cax=cax) # match colorbar with grid size
cbar.ax.tick_params(labelsize=5)
np.savetxt('gaussian_120x120.txt', gaussian)
plt.savefig('gaussian_120x120.png', dpi=300)

# Truncate the gaussian realization into 3 facies according to facies proportions
faciesMap, thresholds = tpg.truncGaussian2facies(gaussian, 0.3, 0.2, 0.5)

# Display the derived facies map with a discrete colorbar
#plt.close()
tpg.discrete_imshow(faciesMap)
np.savetxt('faciesMap_tg_120x120.txt', faciesMap, fmt='%d')
plt.savefig('faciesMap_tg_120x120.png', dpi=300)
plt.show()

