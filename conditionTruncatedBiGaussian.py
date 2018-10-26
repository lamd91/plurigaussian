#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tpg

# Grid dimensions
NX = 120 # number of gridblocks along horizontal axis
NY = 120 # number of gridblocks along vertical axis
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
synFaciesData = np.loadtxt('synFaciesData_from_tpg_REF.txt') # the first and second columns correspond to the x and y coordinates of the data
#nbOfData = synFaciesData.shape[0]

# Assign data to center of nearest gridblock
xcoord_centroids = XX[0, :]
ycoord_centroids = YY[:, 0]
x_data, y_data = tpg.assignDataToNearestCellCentroidCoordinates(xcoord_centroids, ycoord_centroids, synFaciesData) # x and y coordinates of data for conditioning correspond to gridblock centroids
#print(x_data, y_data)

# Get cell coordinates of data
lineIndices_data, colIndices_data = tpg.findDataCellCoordinates(x_data, y_data, XX, YY)
#print(lineIndices_data, colIndices_data)
# faciesMap_ref = np.loadtxt('faciesMap_tpg_ref4HardDataCond.txt')
# The 3rd column of synFaciesData (synFaciesData[:, 2]) correspond to faciesMap_ref[ineIndices_data, colIndices_data]

# Convert facies observation to gaussian pseudo data (Gibbs sampling)
it_max = 200 # total number of iterations
it_st = 100 # iteration at which the distribution is sampled from (should be after the burn-in period; check convergence)
thresholds_g1 = [-0.1] # use the same thresholds as the ones used to obtain the reference
thresholds_g2 = [0] # use the same thresholds as the ones used to obtain the reference
thresholds = thresholds_g1 + thresholds_g2
pseudoData_ini_g1 = tpg.convertFacies2IniPseudoData_tpg_g1(synFaciesData[:, 2], 2)
pseudoData_ini_g2 = tpg.convertFacies2IniPseudoData_tpg_g2(synFaciesData[:, 2], 2)
pseudoData_fin_g1 = tpg.gibbsSampling(pseudoData_ini_g1, synFaciesData[:, 0], synFaciesData[:, 1], thresholds_g1, it_max, it_st)
pseudoData_fin_g2 = tpg.gibbsSampling(pseudoData_ini_g2, synFaciesData[:, 0], synFaciesData[:, 1], thresholds_g2, it_max, it_st)
print(pseudoData_ini_g1)
print(pseudoData_ini_g2)
print(pseudoData_fin_g1)
print(pseudoData_fin_g2)

# Generate/Load existing unconditional gaussian simulations
#gaussian1_uc = tpg.genGaussian2DSim_FFT(120, 120, 0.5, 0.5, 'exponential', 30, 9) 
#gaussian2_uc = tpg.genGaussian2DSim_SGSim_aniso(120, 120, 0.5, 0.5, 'spherical', 10, 0.5, 45)
#np.savetxt('gaussian1_uc.txt', gaussian1_uc)
#np.savetxt('gaussian2_uc.txt', gaussian2_uc)
gaussian1_uc = np.loadtxt('gaussian1_uc.txt')
gaussian2_uc = np.loadtxt('gaussian2_uc.txt')

# Truncate both the unconditional gaussian realizations to one facies realization
facies_uc = tpg.truncBiGaussian23facies(gaussian1_uc, gaussian2_uc, 2, thresholds)
print(synFaciesData[:, 2])
print(facies_uc[lineIndices_data, colIndices_data])
 
# Compute kriging estimate of pseudo data for first realization  
print(pseudoData_fin_g1.shape)
sk_est_data_1, sk_var_data_1, sk_weights_data_1 = tpg.simpleKrig_vector(x_data, y_data, pseudoData_fin_g1, XX.reshape(-1, 1), YY.reshape(-1, 1), 'exponential', 15, 0, 1)
sk_data_grid_1 = sk_est_data_1.reshape(NY, NX)
#print(pseudoData)
#print(sk_data_grid[lineIndices_data, colIndices_data]) # check kriged values at data locations

# Compute kriging estimate of pseudo data for second realization
sk_est_data_2, sk_var_data_2, sk_weights_data_2 = tpg.simpleKrig_vector(x_data, y_data, pseudoData_fin_g2, XX.reshape(-1, 1), YY.reshape(-1, 1), 'spherical', 10, 0, 1)
sk_data_grid_2 = sk_est_data_2.reshape(NY, NX)
#print(pseudoData)
#print(sk_data_grid[lineIndices_data, colIndices_data]) # check kriged values at data locations

# Compute kriging estimate of values simulated by first unconditional realization at data locations
simData_dataLoc_1 = gaussian1_uc[lineIndices_data, colIndices_data]
sk_est_simData_1, sk_var_simData_1, sk_weights_simData_1 = tpg.simpleKrig_vector(x_data, y_data, simData_dataLoc_1, XX.reshape(-1, 1), YY.reshape(-1, 1), 'exponential', 15, 0, 1)
sk_simData_grid_1 = sk_est_simData_1.reshape(NY, NX)
#print(simData_dataLoc)
#print(sk_simData_grid[lineIndices_data, colIndices_data]) # check kriged values at data locations

# Compute kriging estimate of values simulated by second unconditional realization at data locations
simData_dataLoc_2 = gaussian2_uc[lineIndices_data, colIndices_data]
sk_est_simData_2, sk_var_simData_2, sk_weights_simData_2 = tpg.simpleKrig_vector(x_data, y_data, simData_dataLoc_2, XX.reshape(-1, 1), YY.reshape(-1, 1), 'spherical', 10, 0, 1)
sk_simData_grid_2 = sk_est_simData_2.reshape(NY, NX)
#print(simData_dataLoc)
#print(sk_simData_grid[lineIndices_data, colIndices_data]) # check kriged values at data locations

# Condition unconditional gaussian simulation with gaussian pseudo data
gaussian1_c = sk_data_grid_1 + (gaussian1_uc - sk_simData_grid_1)
gaussian2_c = sk_data_grid_2 + (gaussian2_uc - sk_simData_grid_2)
np.savetxt('gaussian1_c.txt', gaussian1_c)
np.savetxt('gaussian2_c.txt', gaussian2_c)

# Truncate conditional gaussian realization to facies
facies_c = tpg.truncBiGaussian23facies(gaussian1_c, gaussian2_c, 2, thresholds)
print(synFaciesData[:, 2])
print(facies_c[lineIndices_data, colIndices_data])

# Display unconditional realization/kriging estimate of data/kriging estimate of values simulated at data locations/conditional simulation/conditional facies realization
plt.close()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))
fig.subplots_adjust(wspace=0.4)
minVal = -3.5
maxVal = 3.5
im1 = ax1.imshow(facies_uc, cmap='rainbow', vmin = np.min(facies_uc)-.5, vmax = np.max(facies_uc)+.5, alpha=0.5, origin='lower')
im2 = ax2.imshow(facies_c, cmap='rainbow', vmin = np.min(facies_c)-.5, vmax = np.max(facies_c)+.5, alpha=0.5, origin='lower')
ax1.set_title('Unconditional facies realization', fontsize=10)
ax2.set_title('Conditional facies realization', fontsize=10)
divider1 = make_axes_locatable(ax1)
divider2 = make_axes_locatable(ax2)
cax1 = divider1.append_axes("right", size="3%", pad=0.2)
cax2 = divider2.append_axes("right", size="3%", pad=0.2)
cbar1 = plt.colorbar(im1, cax=cax1)
cbar2 = plt.colorbar(im2, cax=cax2)

plt.savefig('conditioningTPG.png', dpi=300)

plt.show()


