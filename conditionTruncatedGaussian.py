#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myFunctions import makeFlowParFileForGW
import tpg

# Grid dimensions
NX = 60 # number of gridblocks along horizontal axis
NY = 60 # number of gridblocks along vertical axis
dx = 0.5 # horizontal resolution in meter
dy = 0.5 # vertical resolution in meter

# Centroid coordinates
x_min = dx/2 # x coordinates of first column of elements
x_max = NX*dx - dx/2 # x coordinates of last column of elements
y_min = dy/2 # y coordinates of first bottom line of elements
y_max = NY*dy - dy/2 # y coordinates of first top line of elements

x = np.arange(x_min, x_max+dx, dx) # vector of all x coordinates
y = np.arange(y_min, y_max+dy, dy) # vector of all y coordinates
XX, YY = np.meshgrid(x, y) # store x coordinates in array of shape of the grid

# Load local synthetic facies observations (taken from the reference facies map "faciesMap_ref4HardDataCond.txt") for conditioning
synFaciesData = np.loadtxt('synFaciesData_fromREF.txt') # the first and second columns correspond to the x and y coordinates of the data
#nbOfData = synFaciesData.shape[0]

# Assign data to center of nearest gridblock
xcoord_centroids = XX[0, :]
ycoord_centroids = YY[:, 0]
x_data, y_data = tpg.assignDataToNearestCellCentroidCoordinates(xcoord_centroids, ycoord_centroids, synFaciesData) # x and y coordinates of data for conditioning correspond to gridblock centroids
#print(x_data, y_data)

# Get cell coordinates of data
lineIndices_data, colIndices_data = tpg.findDataCellCoordinates(x_data, y_data, XX, YY)
#print(lineIndices_data, colIndices_data)
# faciesMap_ref = np.loadtxt('faciesMap_ref4HardDataCond.txt')
# The 3rd column of synFaciesData (synFaciesData[:, 2]) correspond to faciesMap_ref[ineIndices_data, colIndices_data]

# Convert facies observation to gaussian pseudo data (Gibbs sampling)
it_max = 100 # total number of iterations
it_st = 50 # iteration at which the distribution is sampled from (should be after the burn-in period; check convergence)
thresholds = [0] # use the same thresholds as the ones used to obtain the reference
#print(pseudoData_ini)
pseudoData = tpg.gibbsSampling(synFaciesData, thresholds, it_max, it_st)

# Generate/Load existing unconditional gaussian simulation
#gaussian_uc = tpg.genGaussianSim_2D(NX, NY, dx, dy, 'spherical', 10) 
#gaussian_uc = tpg.genGaussianSim_2D_FFT(NX, NY, dx, dy, 'exponential', 20, 20) 
#np.savetxt('gaussian_uc.txt', gaussian_uc)
gaussian_uc = np.loadtxt('gaussian_uc.txt')

# Truncate unconditional gaussian realization to facies
facies_uc, props_uc = tpg.truncGaussian2faciesUsingThresholds(gaussian_uc, thresholds)
print(synFaciesData[:, 2])
print(facies_uc[lineIndices_data, colIndices_data])
print(props_uc)
 
# Compute kriging estimate of data
sk_est_data, sk_var_data, sk_weights_data = tpg.simpleKrig_vector(x_data, y_data, pseudoData, XX.reshape(-1, 1), YY.reshape(-1, 1), 'spherical', 10, 0, 1)
sk_data_grid = sk_est_data.reshape(NY, NX)
#print(pseudoData)
#print(sk_data_grid[lineIndices_data, colIndices_data]) # check kriged values at data locations

# Compute kriging estimate of values simulated at data location
simData_dataLoc = gaussian_uc[lineIndices_data, colIndices_data]
sk_est_simData, sk_var_simData, sk_weights_simData = tpg.simpleKrig_vector(x_data, y_data, simData_dataLoc, XX.reshape(-1, 1), YY.reshape(-1, 1), 'spherical', 10, 0, 1)
sk_simData_grid = sk_est_simData.reshape(NY, NX)
#print(simData_dataLoc)
#print(sk_simData_grid[lineIndices_data, colIndices_data]) # check kriged values at data locations

# Condition unconditional gaussian simulation with gaussian pseudo data
gaussian_c = sk_data_grid + (gaussian_uc - sk_simData_grid)
np.savetxt('gaussian_c.txt', gaussian_c)

# Truncate conditional gaussian realization to facies
facies_c, props_c = tpg.truncGaussian2faciesUsingThresholds(gaussian_c, thresholds)
print(synFaciesData[:, 2])
print(facies_c[lineIndices_data, colIndices_data])
print(props_c)

# Display unconditional realization/kriging estimate of data/kriging estimate of values simulated at data locations/conditional simulation/conditional facies realization
plt.close()
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(20,8))
fig.subplots_adjust(wspace=0.4)
minVal = -3.5
maxVal = 3.5
im1 = ax1.imshow(gaussian_uc, vmin=minVal, vmax=maxVal, cmap='rainbow', origin='lower')
im2 = ax2.imshow(facies_uc, cmap='rainbow', vmin = np.min(facies_uc)-.5, vmax = np.max(facies_uc)+.5, alpha=0.5, origin='lower')
im3 = ax3.imshow(sk_est_data.reshape(NY, NX), vmin=minVal, vmax=maxVal, cmap='rainbow', origin='lower')
im4 = ax4.imshow(sk_est_simData.reshape(NY, NX), vmin=minVal, vmax=maxVal, cmap='rainbow', origin='lower')
im5 = ax5.imshow(gaussian_c, vmin=minVal, vmax=maxVal, cmap='rainbow', origin='lower')
im6 = ax6.imshow(facies_c, cmap='rainbow', vmin = np.min(facies_c)-.5, vmax = np.max(facies_c)+.5, alpha=0.5, origin='lower')
ax1.set_title('Unconditional realization', fontsize=10)
ax2.set_title('Unconditional facies realization', fontsize=10)
ax3.set_title('Kriging estimate of data', fontsize=10)
ax4.set_title('Kriging estimate of values\n simulated at data location', fontsize=10)
ax5.set_title('Conditional gaussian realization', fontsize=10)
ax6.set_title('Conditional facies realization', fontsize=10)
divider1 = make_axes_locatable(ax1)
divider2 = make_axes_locatable(ax2)
divider3 = make_axes_locatable(ax3)
divider4 = make_axes_locatable(ax4)
divider5 = make_axes_locatable(ax5)
divider6 = make_axes_locatable(ax6)
cax1 = divider1.append_axes("right", size="3%", pad=0.2)
cax2 = divider2.append_axes("right", size="3%", pad=0.2)
cax3 = divider3.append_axes("right", size="3%", pad=0.2)
cax4 = divider4.append_axes("right", size="3%", pad=0.2)
cax5 = divider5.append_axes("right", size="5%", pad=0.2)
cax6 = divider6.append_axes("right", size="5%", pad=0.2)
cbar1 = plt.colorbar(im1, cax=cax1)
cbar2 = plt.colorbar(im2, cax=cax2)
cbar3 = plt.colorbar(im3, cax=cax3)
cbar4 = plt.colorbar(im4, cax=cax4)
cbar5 = plt.colorbar(im5, cax=cax5)
cbar6 = plt.colorbar(im6, cax=cax6)

plt.savefig('conditioningTG.png', dpi=300)

plt.show()


