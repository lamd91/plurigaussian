#!/usr/bin/python3

# Library containing useful functions for the generation of Truncated Pluri-Gaussian Simulations (TPG)

import numpy as np
from numpy.linalg import inv
from scipy.special import erf
from scipy.optimize import brentq
from math import pi, radians, degrees
from matplotlib import pyplot as plt


def variogram(model, h, sill, range):
	"""
	Provides isotropic variogram models given type of model, vector of distances, sill and range.
	"""

	if model == "gaussian":
		return sill * (1 - np.exp(-3 * (h / range)**2))

	elif model == "exponential":
		return sill * (1 - np.exp(-3 * h / range))

	elif model == "spherical":
		return sill * (3 * h / (2 * range) - 0.5 * (h / range)**3) * (h < range) + sill * (h >= range)


def genGaussianSim_2D(NX, NY, dx, dy, varioType, L):

	"""
	Simulates a continuous gaussian realization N(0,1) using the Sequential Gaussian Simulation (SGSim) method.
	Returns an array of gaussian values.
	
	Parameters
	----------
	NX : integer
		number of gridblocks along x axis

	NY : integer
		number of gridblocks along y axis

	dx : float
		resolution of gridblock along x axis in meters

	dy : float
		resolution of gridblock along y axis in meters
	
	varioType : string
		name of the variogram model: "gaussian", "exponential" or "spherical"

	L : float
		range in meters of exponential variogram model
	
	Returns
	----------
	ndarray 
		numpy array of size NX-1, NY-1
	
	"""

	# Determine the element centroid coordinates of the 2D simulation grid

	x_min = dx/2 # x coordinates of first column of elements
	x_max = NX/dx - dx/2 # x coordinates of last column of elements
	y_min = dy/2 # y coordinates of first bottom line of elements
	y_max = NY/dy - dy/2 # y coordinates of first top line of elements

	x = np.arange(x_min, x_max+dx, dx) # vector of all x coordinates
	y = np.arange(y_min, y_max+dy, dy) # vector of all y coordinates
	XX, YY = np.meshgrid(x, y) # store x coordinates in array of shape of the grid

	XX_flattened = np.reshape(XX, -1) # flattened arrays
	YY_flattened = np.reshape(YY, -1)

	nbElements = NX*NY # total number of gridblocks
	lines_gridCells = np.unravel_index(np.arange(nbElements), (NY, NX))[0] # vector storing the line number of every cell
	col_gridCells = np.unravel_index(np.arange(nbElements), (NY, NX))[1] # vector storing the column number of every cell


	# Define grid filled with values drawn from a uniform distribution between 1e-8 and 1 (excluded)
	GRID_uni = np.random.uniform(1e-8, 1, (NY,NX))

	# Create grid to populate gradually with continuous gaussian values
	GRID_gauss = np.zeros((NY,NX))

	# Define a random path to visit each node of the grid once only

	nbCellsLeftToVisit = nbElements
	xCoord_alreadyVisitedCells = [] # list to store the x-coordinates of the already visited/simulated cells
	yCoord_alreadyVisitedCells = []
	simValues_alreadyVisitedCells = []


	# Variogram parameters
	variogRange = L # range of variogram in meters
	C0 = 1 # data variance set equal to the imposed value 
	mean = 0
	var = 1
	std = var**(1/2)

	# Cell by cell simulation loop 
	while nbCellsLeftToVisit != 0: 
		
		flattenedIndex_visitedCell = int(np.floor(np.random.rand()*nbCellsLeftToVisit)) # index of visited cell in flattened array
		xCoord_visitedCell = XX_flattened[flattenedIndex_visitedCell] # x coordinates of the visited cell
		yCoord_visitedCell = YY_flattened[flattenedIndex_visitedCell]
		line_visitedCell = lines_gridCells[flattenedIndex_visitedCell] 
		col_visitedCell = col_gridCells[flattenedIndex_visitedCell]

		xCoord_alreadyVisitedCells_array = np.asarray(xCoord_alreadyVisitedCells)
		yCoord_alreadyVisitedCells_array = np.asarray(yCoord_alreadyVisitedCells)

		# Look for indices of already simulated cells located inside the square centered at the current simulation cell
		xCoordWindow_east = xCoord_visitedCell + variogRange
		xCoordWindow_west = xCoord_visitedCell - variogRange
		yCoordWindow_north = yCoord_visitedCell + variogRange
		yCoordWindow_south = yCoord_visitedCell - variogRange

		indices_xCoord_alreadyVisitedCells = np.where((xCoord_alreadyVisitedCells_array < xCoordWindow_east) & (xCoord_alreadyVisitedCells_array > xCoordWindow_west))
		indices_yCoord_alreadyVisitedCells = np.where((yCoord_alreadyVisitedCells_array < yCoordWindow_north) & (yCoord_alreadyVisitedCells_array > yCoordWindow_south))
		indices_alreadyVisitedCellsInsideWindow = np.intersect1d(indices_xCoord_alreadyVisitedCells, indices_yCoord_alreadyVisitedCells)

		xCoord_alreadyVisitedCells_withinNeighborhood = [] # list to store the x-coordinates of the already simulated cells within the neighbordhood defined around the simulation cell (cercle of radius equal to the variogRange of the variogram)
		yCoord_alreadyVisitedCells_withinNeighborhood = []
		simValues_alreadyVisitedCells_withinNeighborhood = [] # list of the already simulated values within the neighborhood
		distancesToAlreadyVisitedCells_withinNeighborhood = [] # list containing the distances between the simulation cell and the already simulated ones
		
		for k in indices_alreadyVisitedCellsInsideWindow: # loop over simulated cells in the neighborhood of simulation cell
			dist = ((xCoord_alreadyVisitedCells[k] - xCoord_visitedCell)**2 + (yCoord_alreadyVisitedCells[k] - yCoord_visitedCell)**2)**(1/2) 
			if dist <= variogRange: # if already simulated point is within the cercle centered at simulation cell (neighborhood)
				xCoord_alreadyVisitedCells_withinNeighborhood.append(xCoord_alreadyVisitedCells[k])
				yCoord_alreadyVisitedCells_withinNeighborhood.append(yCoord_alreadyVisitedCells[k])
				simValues_alreadyVisitedCells_withinNeighborhood.append(simValues_alreadyVisitedCells[k])
				distancesToAlreadyVisitedCells_withinNeighborhood.append(dist)
						

		nbOfSimulatedCellsWithinNeighbd = len(xCoord_alreadyVisitedCells_withinNeighborhood) # number of already simulated cells within neighborhood
	#	print(nbOfSimulatedCellsWithinNeighbd)


		if nbOfSimulatedCellsWithinNeighbd == 0: # if no already simulated cells within neighborhood
			m = mean
			sig = std

		else: # Computation of the mean m and the standard deviation sig of the cumulative distribution function
		
			# Define the covariance matrix containing the spatial correlation values estimated between pairs of already simulated points
			C = np.zeros((nbOfSimulatedCellsWithinNeighbd, nbOfSimulatedCellsWithinNeighbd))
	
			for i in np.arange(nbOfSimulatedCellsWithinNeighbd): # fill covariance matrix line by line
				distBetweenPairsOfSimulatedPoints = ((xCoord_alreadyVisitedCells_withinNeighborhood - np.repeat(xCoord_alreadyVisitedCells_withinNeighborhood[i], nbOfSimulatedCellsWithinNeighbd))**2 + (yCoord_alreadyVisitedCells_withinNeighborhood - np.repeat(yCoord_alreadyVisitedCells_withinNeighborhood[i], nbOfSimulatedCellsWithinNeighbd))**2)**(1/2)
				C[i, :] = C0 - variogram(varioType, distBetweenPairsOfSimulatedPoints, C0, variogRange)

			# Define the vector of spatial correlations between the simulation cell and the already simulated points within the neighborhood
			cov_vector = C0 - variogram(varioType, np.asarray(distancesToAlreadyVisitedCells_withinNeighborhood), C0, variogRange)

			# Computation of the kriging weights lambdas	
			lambdas = np.dot(inv(C), cov_vector)
			
			# Computation of the mean m of the cdf using the simple kriging mean
			m = mean + np.sum(lambdas*(np.asarray(simValues_alreadyVisitedCells_withinNeighborhood) - mean))
		
			# Computation of the standard deviation sig of the cdf using the simple kriging variance
			var = C0 - np.sum(lambdas*(C0 - variogram(varioType, np.asarray(distancesToAlreadyVisitedCells_withinNeighborhood), C0, variogRange)))
			sig = var**(1/2)	


		# Compute the antecedent of the uniform value by the normal cdf at the visited cell, i.e. find the root of the function corresponding to the normal cdf minus the uniform value

		def conditionalCdf_minusUniform(x):
			# function equal to the normal cdf minus the uniform value (in order after to compute its root using the Brent method)
			y = 0.5*(1+erf((x-m)/((2)**(1/2)*sig))) - GRID_uni[line_visitedCell, col_visitedCell] 
			return y

		gaussianValue = brentq(conditionalCdf_minusUniform, -5, 5) # finds the root of the monotonic function f within the interval [-5, 5]
			
		# Update grid of gaussian values	
		GRID_gauss[line_visitedCell, col_visitedCell] = gaussianValue
		simValues_alreadyVisitedCells.append(gaussianValue)

		# Update lists of already simulated points with the last visited point
		xCoord_alreadyVisitedCells.append(xCoord_visitedCell)
		yCoord_alreadyVisitedCells.append(yCoord_visitedCell)

		# Remove visited cell from vectors in order to not visit the cells more than once
		XX_flattened = np.delete(XX_flattened, flattenedIndex_visitedCell)
		YY_flattened = np.delete(YY_flattened, flattenedIndex_visitedCell)	
		lines_gridCells = np.delete(lines_gridCells, flattenedIndex_visitedCell)
		col_gridCells = np.delete(col_gridCells, flattenedIndex_visitedCell)
		
		nbCellsLeftToVisit = nbCellsLeftToVisit-1 

	return GRID_gauss


def truncGaussian2facies(gaussianGrid, p1, p2, p3):

	"""
	Truncates the continuous gaussian realization into 3 facies according to thresholds calculated based on known facies proportions.
	Returns an array filled with values 1, 2 or 3 corresponding to either of the 3 facies 
	
	Parameters
	----------
	gaussianGrid : ndarray
		numpy array filled with continuous gaussian values

	p1 : float
		proportion of facies of value 1
	p2 : float
		proportion of facies of value 2
	p3 : float
		proportion of facies of value 3

	Returns
	----------
	ndarray
		numpy array of same size as gaussianGrid

	"""

	# Facies proportions (assumed stationary)
	prop_facies1 = p1 # proportion of facies of value 1
	prop_facies2 = p2
	prop_facies3 = p3

	# Derive the thresolds given the facies proportions
	NY = gaussianGrid.shape[0] # number of gridblocks along y axis
	NX = gaussianGrid.shape[1]
	nbElements = NX*NY 
	nbElements4facies1 = int(np.round(prop_facies1*nbElements)) # number of gridblocks to be assigned with facies 1
	nbElements4facies1And2 = int(np.round((prop_facies1+prop_facies2)*nbElements)) # number of gridblocks to be assigned with facies 1 or 2
	gaussianGrid_flattened_sorted = np.sort(np.reshape(gaussianGrid, -1))
	threshold_S1 = gaussianGrid_flattened_sorted[nbElements4facies1] # threshold gaussian value between facies 1 and 2
	threshold_S2 = gaussianGrid_flattened_sorted[nbElements4facies1And2] # threshold gaussian value between facies 2 and 3

	# Assign facies values according to derived thresholds
	faciesGrid = np.ones((NY, NX))*2 # initialize facies grid with facies 2
	faciesGrid[np.where(gaussianGrid <= threshold_S1)[0], np.where(gaussianGrid <= threshold_S1)[1]] = 1
	faciesGrid[np.where(gaussianGrid > threshold_S2)[0], np.where(gaussianGrid > threshold_S2)[1]] = 3

	return faciesGrid


def discrete_imshow(faciesGrid):

	"""
	Displays facies map using a discrete colormap with ticks at the centre of each color.
	
	Parameters
	----------
	faciesGrid : ndarray
		numpy array filled with discrete values

	"""
	
	plt.close()
	# Get discrete colormap
	cmap = plt.get_cmap('rainbow', np.max(faciesGrid)-np.min(faciesGrid)+1)
	# Set limits .5 outside true variogRange
	im = plt.imshow(faciesGrid, cmap=cmap, vmin = np.min(faciesGrid)-.5, vmax = np.max(faciesGrid)+.5, alpha=0.5)
	# Tell the colorbar to tick at integers
	cax = plt.colorbar(im, ticks=np.arange(np.min(faciesGrid),np.max(faciesGrid)+1))
	plt.show()


def segment_intersection(line1, line2, xmin, xmax, ymin, ymax):

	"""
	Function which computes the intersection of 2 segments or return false when the 2 segments don't intersect.

	Parameters
	----------


	Returns
	-------
	
	
	"""

	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	def det(a, b):
	    return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	if div == 0:
		return False # segments do not intersect

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div

	if x > xmax or x < xmin or y > ymax or y < ymin:
		return False # segments do not intersect within domain
		
	return x, y



class truncLines():

	def __init__(self):

		nbLines = 2 # number of truncation lines set by default
		self.angles = np.array([3.0087741, 7.73074376])
		self.dist2origin = np.array([0.55325674, 0.26806778])

		# Define horizontal and vertical axes of truncation map
		g1 = np.linspace(-4, 4, 2)
		g2 = np.linspace(-4, 4, 2)

		# Define 3 random segments each defined by a rotation angle and a distance to the origin
		plt.close()
		plt.figure()

		lines = np.zeros((nbLines, g1.shape[0])) # array that stores the 2 endpoints of the 3 random lines to be generated
		rotationAngles = np.zeros(nbLines) # vector that stores the rotation angle of each random line to be generated
		distancesToOrigin = np.zeros(nbLines) # vector that stores the rotation angle of each random line to be generated

		# For each line
		colors = ['r', 'g']

		for i in np.arange(nbLines):
			
#			rotationAngles[i] = np.random.uniform(pi/2, pi/2+2*pi)
#			distancesToOrigin[i] = np.random.uniform(0, 1)
			rotationAngles[i] = self.angles[i]
			distancesToOrigin[i] = self.dist2origin[i]
			lines[i, :] = np.tan(rotationAngles[i]-pi/2)*(g1 - distancesToOrigin[i]/np.cos(rotationAngles[i]))

			plt.plot(g1, lines[i, :], color=colors[i], alpha=0.5)


		# Plot the 3 generated lines
		plt.xlim(-4, 4)
		plt.ylim(-4, 4)
		plt.axhline(color='k', alpha=0.8, linestyle='--', linewidth=0.2)
		plt.axvline(color='k', alpha=0.8, linestyle='--', linewidth=0.2)
		plt.savefig('truncRules.png')
		#plt.show()

		# Compute the intersections between the 3 lines 
		inter_RG = segment_intersection(([g1[0],lines[0,0]], [g1[1], lines[0,1]]), ([g1[0], lines[1,0]], [g1[1], lines[1,1]]), -4, 4, -4, 4) # tests intersection between red and green lines

		# While there are no 3 intersection points, regenerate the 3 lines
		while segment_intersection(([g1[0],lines[0,0]], [g1[1], lines[0,1]]), ([g1[0], lines[1,0]], [g1[1], lines[1,1]]), -4, 4,-4, 4) == False:

			for i in np.arange(nbLines):
				
			#	rotationAngles[i] = np.random.uniform(pi/2, pi/2+2*pi)
			#	distancesToOrigin[i] = np.random.uniform(0, 1)
				rotationAngles[i] = self.angles[i]
				distancesToOrigin[i] = self.dist2origin[i]
				lines[i, :] = np.tan(rotationAngles[i]-pi/2)*(g1 - distancesToOrigin[i]/np.cos(rotationAngles[i]))
				plt.plot(g1, lines[i, :], color=colors[i], alpha=0.5)

		
			# Plot the 3 intersecting lines
			plt.xlim(-4, 4)
			plt.ylim(-4, 4)
			plt.axhline(color='k', alpha=0.8, linestyle='--', linewidth=0.2)
			plt.axvline(color='k', alpha=0.8, linestyle='--', linewidth=0.2)
			plt.savefig('truncRules.png')
#			plt.show()

			# Compute the 3 intersections points
			inter_RG = segment_intersection(([g1[0],lines[0,0]], [g1[1], lines[0,1]]), ([g1[0], lines[1,0]], [g1[1], lines[1,1]]), -4, 4, -4, 4) # between red and green lines


#		# Print the coordinates of the 3 intersections points
#		print("Coordinates of intersection points:")
#		print(inter_RG[0], inter_RG[1])	

def thresholdLineEq(r, teta, x):

	y = np.tan(teta-pi/2)*(x - r/np.cos(teta)) 
	
	return y
	

def truncBiGaussian2facies(g1, g2, lines):

	"""
	"""
	faciesGrid = g2
	faciesGrid[np.where((g2 < thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & (g2 > thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[0], np.where((g2 < thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & (g2 > thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[1]] = 1
	faciesGrid[np.where((g2 > thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & (g2 > thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[0], np.where((g2 > thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & (g2 > thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[1]] = 2
	faciesGrid[np.where((g2 > thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & (g2 < thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[0], np.where((g2 > thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & (g2 < thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[1]] = 3	
	faciesGrid[np.where((g2 < thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & (g2 < thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[0], np.where((g2 < thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & (g2 < thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[1]] = 4	

	return faciesGrid
	



 
