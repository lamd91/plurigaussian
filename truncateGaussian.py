#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myFunctions import makeFlowParFileForGW
import tpg


# Simulate one continuous gaussian realization

#isoModel = tpg.model_isotropic('spherical', 1, 10)
#gaussian = tpg.genGaussian2DSim_SGSim_iso(30, 30, 1, 1, isoModel)

anisoModel = tpg.model_anisotropic('exponential', 1, 10, 1, 0)
gaussian = tpg.genGaussian2DSim_SGSim_aniso(30, 30, 1, 1, anisoModel)

# Display generated gaussian realization
plt.close()
fig, ax = plt.subplots()
im = plt.imshow(gaussian, origin='lower', cmap='rainbow')
divider = make_axes_locatable(ax) 
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(im, cax=cax) # match colorbar with grid size
cbar.ax.tick_params(labelsize=5)
plt.savefig('gaussian.png', dpi=300)

# Truncate the gaussian realization into 3 facies according to facies proportions
faciesMap, thresholds = tpg.truncGaussian2facies(gaussian, 0.3, 0.2, 0.5)

# Display the derived facies map with a discrete colorbar
plt.close()
tpg.discrete_imshow(faciesMap)
plt.savefig('faciesMap.png', dpi=300)

