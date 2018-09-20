#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tpg

# Simulate one continuous gaussian realization
gaussian = tpg.genGaussianSim_2D_aniso(120, 120, 0.5, 0.5, 'exponential', 10, 0.5, 45)
#gaussian = tpg.genGaussianSim_2D_aniso(120, 120, 0.5, 0.5, 'spherical', 10, 0.5, 135)

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
faciesMap = tpg.truncGaussian2facies(gaussian, 0.3, 0.2, 0.5)

# Display the derived facies map with a discrete colorbar
plt.close()
tpg.discrete_imshow(faciesMap)
plt.savefig('faciesMap.png', dpi=300)


