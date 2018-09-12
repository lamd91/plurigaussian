#!/usr/bin/python3

import tpg
import numpy as np
from matplotlib import pyplot as plt

# Simulate two continuous gaussian realizations using different variogram models
gaussian1 = tpg.genGaussianSim_2D(30, 30, 1, 1, 'exponential', 10)
gaussian2 = tpg.genGaussianSim_2D(30, 30, 1, 1, 'gaussian', 5)

## Display each generated gaussian realizations
#plt.imshow(gaussian1, cmap='rainbow', aspect='auto')
#plt.colorbar()
#plt.savefig('gaussian1.png', dpi=300)
#plt.close()
#plt.imshow(gaussian2, cmap='rainbow', aspect='auto')
#plt.colorbar()
#plt.savefig('gaussian2.png', dpi=300)
#plt.close()

# Truncate according to given rock type rule 
thresholdLines = tpg.truncLines() # generate two random threshold lines 
#A = tpg.thresholdLineEq(thresoldLines.dist2origin[0], thresoldLines.angles[0], gaussian1)
#print(A)
faciesMap = tpg.truncBiGaussian2facies(gaussian1, gaussian2, thresholdLines)

# Display the derived facies map with a discrete colorbar 
tpg.discrete_imshow(faciesMap)
plt.savefig('faciesMap_afterTPG.png', dpi=300)
#plt.show()
