#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myFunctions import makeFlowParFileForGW
import tpg
import backup_tpg
import covModel
import grf

# Simulate one continuous gaussian realization

# - Option 1: Using SGSim 
#model = backup_tpg.model_isotropic('exponential', 1, 10)
#gaussian = backup_tpg.genGaussian2DSim_SGSim_iso(60, 60, 0.5, 0.5, model)
model = tpg.model('exponential', 1, 10, 0.5, 60)
gaussian = tpg.genGaussian2DSim_SGSim(120, 120, 0.5, 0.5, model, 0)

## - Option 2: Using FFT
#
## Define grid
#nx, ny = 500, 50
#dx, dy = 10., 10.
#ox, oy = 0., 0.
#
#dimension = [nx, ny]
#spacing = [dx, dy]
#origin = [ox, oy]
#
## Define covariance model
#cov_model = covModel.CovModel2D(elem=[
#                ('exponential', {'w':1, 'r':[100*dx, 10*dy]}), # elementary contribution
#                ('nugget', {'w':0})                   # elementary contribution
#                ], alpha=-5, name='')
#
## Get covariance function and range
#cov_fun = cov_model.func()
#
## Define mean and variance of GRF
#mean = 0
#var = 1
#
## Define hard data
##x = np.array([[ 10.,  20.], # 1st point
##              [ 50.,  40.], # 2nd point
##              [ 20., 150.], # 3rd point
##              [200., 210.]]) # 4th point
##v = [ 8.,  9.,   8.,  12.] # values
#x, v = None, None
#
## Set minimal extension according to the size of the grid and the range
#extensionMin = [grf.extension_min(r, n, s) for r, n, s in zip(cov_model.rxy(), dimension, spacing)]
#
## Generate GRF
#gaussian = grf.grf2D(cov_fun, dimension, spacing, origin=origin,
#            nreal=1, mean=mean, var = var,
#            x=x, v=v,
#            method=3, conditioningMethod=2,
#            extensionMin=extensionMin).reshape(ny, nx) # grf: (nreal,ny,nx) array

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
#plt.close()
tpg.discrete_imshow(faciesMap)
plt.savefig('faciesMap.png', dpi=300)
plt.show()

