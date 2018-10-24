#!/usr/bin/python3

# Library containing useful functions for the generation of Truncated Pluri-Gaussian Simulations (TPG)

import numpy as np
from numpy.linalg import inv, norm, lstsq
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import solve
from scipy.special import erf
from scipy.optimize import brentq
from math import pi, radians, degrees
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def simpleKrig_vector(x, y, v, xi, yi, varioType, L, mu, C0):
	"""
	Simple kriging implementation
	Arguments:
	x, y, v: data points
	xi, yi: the points where kriging interpolation is requested
	covmodel: covariance model function
	mu: mean
	Results:
	v_est : array of estimated values at locations (xi,yi)
	v_var : array of kriging variances at locations (xi,yi)
	"""

	n = x.shape[0]
	if n==0:
		d = 0;
	else:
		X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
		d = squareform(pdist(X))
	C = C0 - variogram(varioType, d, C0, L)
	c = np.transpose(C0 - variogram(varioType, np.sqrt((xi-x)**2 + (yi-y)**2), C0, L))
	l = solve(C, c)

	v_est = np.dot(np.transpose(l), v - mu) + mu
	v_var = C0 - np.diag(np.dot(np.transpose(l), c))

	return v_est, v_var, l


def simpleKrig(x, y, v, xi, yi, varioType, L, mu, C0):
	"""
	Simple kriging implementation
	Arguments:
	x, y, v: data points
	xi, yi: the point where kriging interpolation is requested
	covmodel: covariance model function
	mu: mean
	Results:
	v_est : array of estimated values at locations (xi,yi)
	v_var : array of kriging variances at locations (xi,yi)
	"""

	n = x.shape[0]
	if n==0:
		d = 0;
	else:
		X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
		d = squareform(pdist(X))
	C = C0 - variogram(varioType, d, C0, L);
	c = C0 - variogram(varioType, np.sqrt((xi-x)**2 + (yi-y)**2), C0, L)
	l = solve(C,c)

	v_est = np.sum(l*(v - mu)) + mu
	v_var = C0 - np.sum(l*c)

	return v_est, v_var, l



def variogram(model, h, sill, L):
	"""
	Provides isotropic variogram models given type of model, vector of distances, sill and range.
	"""

	if model == "gaussian":
		return sill*(1-np.exp(-(3**(1/2)*h/L)**2))

	elif model == "exponential":
		return sill * (1 - np.exp(-3 * np.absolute(h) / L))

	elif model == "spherical":
		return sill * (3 * h / (2 * L) - 0.5 * (h / L)**3) * (h < L) + sill * (h >= L)


def variogram_aniso(model, h_x, h_y, sill, range_max, aniso_ratio, angle):
	"""
	Provides anisotropic variogram models given type of model, vector of horizontal component of distances, 
vector of vertical component of distances,  sill, maximum range, anisotropy ratio, counterclockwise rotation angle.
	"""

	R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])	# rotation matrix (angles are measured counterclockwise from east)
	T = np.diag(np.array([1/range_max, 1/(range_max*aniso_ratio)])) # reduced distance scaling matrix

	h = norm(np.transpose(np.hstack((h_x, h_y))), axis=0)
	h_reduced = norm(np.dot(T, np.dot(R, np.transpose(np.hstack((h_x, h_y))))), 2, axis=0) # equal to ratio h/range
	range = h*range_max*range_max*aniso_ratio/norm(np.dot(np.diag(np.array([range_max*aniso_ratio, range_max])), 
np.dot(R, np.transpose(np.hstack((h_x, h_y))))), axis=0) 

	if model == "gaussian":
		return sill * (1 - np.exp(-(3**(1/2) * h_reduced)**2))

	elif model == "exponential":
		return sill * (1 - np.exp(-3 * h_reduced))

	elif model == "spherical":
		return sill * (3/2 * h_reduced - 0.5 * h_reduced**3) * (h < range) + sill * (h >= range)


def genGaussianSim_2D_FFT(NX, NY, mean, var, varioType, range_x, range_y):

	# Modified after Gregoire Mariethoz, 2010. Original code from Olaf Cirpka, 2003.
	# Description:
	# Generates one unconditional realization of a multiGaussian random function using FFT.
	# Output arguments:
	# grid: the grid containing the generated multiGaussian variable.
	# Input arguments:
	# x, y, z: the size of the simulation grid.
	# mu: mean of the simulation.
	# sigma2: variance of the simulation.
	# varioType: covariance model. 'exponential'=exponential covariance. 'gaussian'=Gaussian covariance.
	# lx, ly, lz: correlation lengths. The units are in number of grid nodes.

	## Input arguments:
	x = NX; y = NY; z = 1 # grid of size x*y*z
	mu = mean; sigma2 = var**(1/2) # mean and variance of standard normal distribution
	lx = range_x; ly = range_y; lz = 1 # correlation lengths

	## Generate grid
	zz,yy,xx = np.mgrid[0:z,0:y,0:x]
	nx = xx.shape[2]; ny = xx.shape[1]; nz = xx.shape[0]

	xc = (xx[0,0,-1]-xx[0,0,0])/2 # coordinates of the center of the grid
	yc = (yy[0,-1,0]-yy[0,0,0])/2
	zc = (zz[0,-1,0]-zz[0,0,0])/2

	lx = lx/3; ly = ly/3; lz = lz/3
	h = (((xx-xc)/lx)**2+((yy-yc)/ly)**2+((zz-zc)/lz)**2)**(1/2) # distance to the centre

	if varioType == 'exponential' :
		c = np.exp(-np.absolute(h))*sigma2 # exponential covariance
	elif varioType == 'gaussian' :
		c = np.exp(-h**2)*sigma2 # gaussian covariance

	grid = np.fft.fftn(np.fft.fftshift(c))/(nx*ny*nz)
	grid = np.absolute(grid); grid[0,0,0] = 0
	ran = np.multiply(grid**(1/2),np.exp(1j*np.angle(np.fft.fftn(np.random.rand(nz,ny,nx)))))
	#ran = grid**(1/2)*np.exp(1j*np.angle(np.fft.fftn(np.random.rand(nz,ny,nx))))
	grid = np.real(np.fft.ifftn(ran*nx*ny*nz))

	#gaussian = np.reshape(grid.squeeze(), (nx*ny,1))
	gaussian = np.reshape(grid.squeeze(), (y, x))
	np.mean(gaussian)
	np.std(gaussian)

	return gaussian


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
		range in meters of variogram model
	
	Returns
	-------
	ndarray 
		numpy array of size NX-1, NY-1	
	"""
	
	# Determine the element centroid coordinates of the 2D simulation grid

	x_min = dx/2 # x coordinates of first column of elements
	x_max = NX*dx - dx/2 # x coordinates of last column of elements
	y_min = dy/2 # y coordinates of first bottom line of elements
	y_max = NY*dy - dy/2 # y coordinates of first top line of elements

	x = np.arange(x_min, x_max+dx, dx) # vector of all x coordinates
	y = np.arange(y_min, y_max+dy, dy) # vector of all y coordinates
	XX, YY = np.meshgrid(x, y) # store x coordinates in array of shape of the grid

	XX_flattened = np.reshape(XX, -1) # flattened arrays
	YY_flattened = np.reshape(YY, -1)

	nbElements = NX*NY # total number of gridblocks
	lines_gridCells = np.unravel_index(np.arange(nbElements), (NY, NX))[0] # vector storing the line number of every cell
	col_gridCells = np.unravel_index(np.arange(nbElements), (NY, NX))[1] # vector storing the column number of every cell


	# Define grid filled with values drawn from a uniform distribution between 1e-8 and 1 (excluded)
	GRID_uni = np.random.uniform(1e-8, 1, (NY, NX))

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

		indices_xCoord_alreadyVisitedCells = np.where((xCoord_alreadyVisitedCells_array < xCoordWindow_east) &
 (xCoord_alreadyVisitedCells_array > xCoordWindow_west))
		indices_yCoord_alreadyVisitedCells = np.where((yCoord_alreadyVisitedCells_array < yCoordWindow_north) &
 (yCoord_alreadyVisitedCells_array > yCoordWindow_south))
		indices_alreadyVisitedCellsInsideWindow = np.intersect1d(indices_xCoord_alreadyVisitedCells, 
indices_yCoord_alreadyVisitedCells)

		xCoord_alreadyVisitedCells_withinNeighborhood = [] # list to store the x-coordinates of the already simulated cells within the neighbordhood defined around the simulation cell (cercle of radius equal to the variogRange of the variogram)
		yCoord_alreadyVisitedCells_withinNeighborhood = []
		simValues_alreadyVisitedCells_withinNeighborhood = [] # list of the already simulated values within the neighborhood
		distancesToAlreadyVisitedCells_withinNeighborhood = [] # list containing the distances between the simulation cell and the already simulated ones
		
		for k in indices_alreadyVisitedCellsInsideWindow: # loop over simulated cells in the neighborhood of simulation cell
			dist = ((xCoord_alreadyVisitedCells[k] - xCoord_visitedCell)**2 + 
(yCoord_alreadyVisitedCells[k] - yCoord_visitedCell)**2)**(1/2) 
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
				distBetweenPairsOfSimulatedPoints = ((xCoord_alreadyVisitedCells_withinNeighborhood - 
					np.repeat(xCoord_alreadyVisitedCells_withinNeighborhood[i], nbOfSimulatedCellsWithinNeighbd))**2 + 
					(yCoord_alreadyVisitedCells_withinNeighborhood - 
					np.repeat(yCoord_alreadyVisitedCells_withinNeighborhood[i], nbOfSimulatedCellsWithinNeighbd))**2)**(1/2)
				C[i, :] = C0 - variogram(varioType, distBetweenPairsOfSimulatedPoints, C0, variogRange)

			# Define the vector of spatial correlations between the simulation cell and the already simulated points within the neighborhood
			cov_vector = C0 - variogram(varioType, np.asarray(distancesToAlreadyVisitedCells_withinNeighborhood), C0, 
				variogRange)

			# Computation of the kriging weights lambdas	
#			lambdas = np.dot(inv(C), cov_vector)
			lambdas = lstsq(C, cov_vector)[0]
#			print(lambdas)
			
			# Computation of the mean m of the cdf using the simple kriging mean
			m = mean + np.sum(lambdas*(np.asarray(simValues_alreadyVisitedCells_withinNeighborhood) - mean))
		
			# Computation of the standard deviation sig of the cdf using the simple kriging variance
			var = C0 - np.sum(lambdas*(C0 - variogram(varioType, 
				np.asarray(distancesToAlreadyVisitedCells_withinNeighborhood), C0, variogRange)))
			sig = var**(1/2)	


		# Compute the antecedent of the uniform value by the normal cdf at the visited cell, i.e. find the root of the function corresponding to the normal cdf minus the uniform value

		def conditionalCdf_minusUniform(x):
			# function equal to the normal cdf minus the uniform value (in order after to compute its root using the Brent method)
			y = 0.5*(1+erf((x-m)/((2)**(1/2)*sig))) - GRID_uni[line_visitedCell, col_visitedCell] 
			return y

#		print(m, var)
#		print(conditionalCdf_minusUniform(-5), conditionalCdf_minusUniform(5))
#		print('\n')
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
		print(nbCellsLeftToVisit) 

	return GRID_gauss


def genGaussianSim_2D_aniso(NX, NY, dx, dy, varioType, L_max, aniso_ratio, angle_degrees):
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

	L_max : float
		maximum range in meters of variogram model corresponding to the major axis of ellipse
	
	aniso_ratio : float
		anisotropy ratio such that L_max * aniso_ratio is equal to the minor axis of ellipse

	angle_degrees : float 
		counterclockwise angle of rotation of ellipse, in degrees, from the east direction of cartesian
 coordinate system

	Returns
	-------
	ndarray 
		numpy array of size NX-1, NY-1	
	"""
	
	# Determine the element centroid coordinates of the 2D simulation grid

	x_min = dx/2 # x coordinates of first column of elements
	x_max = NX*dx - dx/2 # x coordinates of last column of elements
	y_min = dy/2 # y coordinates of first bottom line of elements
	y_max = NY*dy - dy/2 # y coordinates of first top line of elements

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
	variogRange_max = L_max # range of variogram in meters
	variogRange_min = L_max*aniso_ratio
	angle = radians(angle_degrees)
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
		xCoordWindow_east = xCoord_visitedCell + variogRange_max
		xCoordWindow_west = xCoord_visitedCell - variogRange_max
		yCoordWindow_north = yCoord_visitedCell + variogRange_max
		yCoordWindow_south = yCoord_visitedCell - variogRange_max

		indices_xCoord_alreadyVisitedCells = np.where((xCoord_alreadyVisitedCells_array < xCoordWindow_east) & 
			(xCoord_alreadyVisitedCells_array > xCoordWindow_west))
		indices_yCoord_alreadyVisitedCells = np.where((yCoord_alreadyVisitedCells_array < yCoordWindow_north) & 
			(yCoord_alreadyVisitedCells_array > yCoordWindow_south))
		indices_alreadyVisitedCellsInsideWindow = np.intersect1d(indices_xCoord_alreadyVisitedCells, 
			indices_yCoord_alreadyVisitedCells)

		xCoord_alreadyVisitedCells_withinNeighborhood = [] # list to store the x-coordinates of the already simulated cells within the neighbordhood defined around the simulation cell (cercle of radius equal to the variogRange of the variogram)
		yCoord_alreadyVisitedCells_withinNeighborhood = []
		simValues_alreadyVisitedCells_withinNeighborhood = [] # list of the already simulated values within the neighborhood
		distToAlreadyVisitedCells_withinNeighborhood_x = [] # list containing the distances between the simulation cell and the already simulated ones
		distToAlreadyVisitedCells_withinNeighborhood_y = [] # list containing the distances between the simulation cell and the already simulated ones
		
		for k in indices_alreadyVisitedCellsInsideWindow: # loop over simulated cells in the neighborhood of simulation cell
			D = (((xCoord_alreadyVisitedCells[k] - xCoord_visitedCell)*np.cos(angle) + (yCoord_alreadyVisitedCells[k] - yCoord_visitedCell)*np.sin(angle))/variogRange_max)**2 + ((-(xCoord_alreadyVisitedCells[k] - xCoord_visitedCell)*np.sin(angle) + (yCoord_alreadyVisitedCells[k] - yCoord_visitedCell)*np.cos(angle))/variogRange_min)**2
			if D <= 1: # if already simulated point is within the cercle centered at simulation cell (neighborhood)
				xCoord_alreadyVisitedCells_withinNeighborhood.append(xCoord_alreadyVisitedCells[k])
				yCoord_alreadyVisitedCells_withinNeighborhood.append(yCoord_alreadyVisitedCells[k])
				simValues_alreadyVisitedCells_withinNeighborhood.append(simValues_alreadyVisitedCells[k])
				distToAlreadyVisitedCells_withinNeighborhood_x.append(xCoord_alreadyVisitedCells[k] - xCoord_visitedCell)
				distToAlreadyVisitedCells_withinNeighborhood_y.append(yCoord_alreadyVisitedCells[k] - yCoord_visitedCell)
						

		nbOfSimulatedCellsWithinNeighbd = len(xCoord_alreadyVisitedCells_withinNeighborhood) # number of already simulated cells within neighborhood
	#	print(nbOfSimulatedCellsWithinNeighbd)


		if nbOfSimulatedCellsWithinNeighbd == 0: # if no already simulated cells within neighborhood
			m = mean
			sig = std

		else: # Computation of the mean m and the standard deviation sig of the cumulative distribution function
		
			# Define the covariance matrix containing the spatial correlation values estimated between pairs of already simulated points
			C = np.zeros((nbOfSimulatedCellsWithinNeighbd, nbOfSimulatedCellsWithinNeighbd))
	
			for i in np.arange(nbOfSimulatedCellsWithinNeighbd): # fill covariance matrix line by line
				distBetweenPairsOfSimulatedPoints_x = xCoord_alreadyVisitedCells_withinNeighborhood - np.repeat(xCoord_alreadyVisitedCells_withinNeighborhood[i], nbOfSimulatedCellsWithinNeighbd)
				distBetweenPairsOfSimulatedPoints_y = yCoord_alreadyVisitedCells_withinNeighborhood - np.repeat(yCoord_alreadyVisitedCells_withinNeighborhood[i], nbOfSimulatedCellsWithinNeighbd)
				C[i, :] = C0 - variogram_aniso(varioType, distBetweenPairsOfSimulatedPoints_x.reshape(-1,1), distBetweenPairsOfSimulatedPoints_y.reshape(-1,1), C0, variogRange_max, aniso_ratio, angle)

	
			# Define the vector of spatial correlations between the simulation cell and the already simulated points within the neighborhood			

			cov_vector = C0 - variogram_aniso(varioType, 
				np.asarray(distToAlreadyVisitedCells_withinNeighborhood_x).reshape(-1,1), 
				np.asarray(distToAlreadyVisitedCells_withinNeighborhood_y).reshape(-1,1), 
				C0, variogRange_max, aniso_ratio, angle)

			# Computation of the kriging weights lambdas	
#			lambdas = np.dot(inv(C), cov_vector)
			lambdas = lstsq(C, cov_vector)[0]
#			print(lambdas.shape)
			
			# Computation of the mean m of the cdf using the simple kriging mean
			m = mean + np.sum(lambdas*(np.asarray(simValues_alreadyVisitedCells_withinNeighborhood) - mean))
		
			# Computation of the standard deviation sig of the cdf using the simple kriging variance
			var = C0 - np.sum(lambdas*(C0 - variogram_aniso(varioType, 
				np.asarray(distToAlreadyVisitedCells_withinNeighborhood_x).reshape(-1,1), 
				np.asarray(distToAlreadyVisitedCells_withinNeighborhood_y).reshape(-1,1), C0, variogRange_max, 
				aniso_ratio, angle)))
			sig = var**(1/2)	


		# Compute the antecedent of the uniform value by the normal cdf at the visited cell, i.e. find the root of the function corresponding to the normal cdf minus the uniform value

		def conditionalCdf_minusUniform(x):
			# function equal to the normal cdf minus the uniform value (in order after to compute its root using the Brent method)
			y = 0.5*(1+erf((x-m)/((2)**(1/2)*sig))) - GRID_uni[line_visitedCell, col_visitedCell] 
			return y

#		print(m, var)
#		print(conditionalCdf_minusUniform(-5), conditionalCdf_minusUniform(5)) # should be of different sign
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
		print(nbCellsLeftToVisit) 

	return GRID_gauss


def truncGaussian2facies(gaussianGrid, p1, p2, p3):
	"""
	Truncates the continuous gaussian realization into 3 facies according to thresholds calculated based on 
known facies proportions.
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
	-------
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
	thresholds = [threshold_S1, threshold_S2]

	# Assign facies values according to derived thresholds
	faciesGrid = np.ones((NY, NX))*2 # initialize facies grid with facies 2
	faciesGrid[np.where(gaussianGrid <= threshold_S1)[0], np.where(gaussianGrid <= threshold_S1)[1]] = 1
	faciesGrid[np.where(gaussianGrid > threshold_S2)[0], np.where(gaussianGrid > threshold_S2)[1]] = 3

	# Compute proportion of each facies
	prop_facies1 = faciesGrid.reshape(-1).tolist().count(1)/faciesGrid.size*100
	prop_facies2 = faciesGrid.reshape(-1).tolist().count(2)/faciesGrid.size*100
	prop_facies3 = faciesGrid.reshape(-1).tolist().count(3)/faciesGrid.size*100
	print('Proportion of facies 1: %d%% ' % prop_facies1)
	print('Proportion of facies 2: %d%% ' % prop_facies2)
	print('Proportion of facies 3: %d%% ' % prop_facies3)

	return faciesGrid, thresholds


def discrete_imshow(faciesGrid):
	"""
	Displays facies map using a discrete colormap with ticks at the centre of each color.
	
	Parameters
	----------
	faciesGrid : ndarray
		numpy array filled with discrete values
	"""
	
	fig, ax = plt.subplots()	

	# Get discrete colormap
	cmap = plt.get_cmap('rainbow', np.max(faciesGrid)-np.min(faciesGrid)+1)
	# Set limits minus and plus 0.5 outside true range
	im = plt.imshow(faciesGrid, origin='lower', cmap=cmap, vmin = np.min(faciesGrid)-.5, vmax = np.max(faciesGrid)+.5, alpha=0.5)
	# Match colorbar with grid size
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="3%", pad=0.2)
	# Display colorbar with ticks at discrete values
	cbar = plt.colorbar(im, cax=cax, ticks=np.arange(np.min(faciesGrid),np.max(faciesGrid)+1))
	cbar.ax.tick_params(labelsize=5)


def segment_intersection(line1, line2, xmin, xmax, ymin, ymax):
	"""
	Function which returns the intersection point coordinates of 2 intersecting lines or return false when
 they do not intersect within the domain delimited

	Parameters
	----------
	line1 : tuple of 2 values
		each value is a list of 2 elements corresponding to the x and y coordinates of an endpoint of 
line1 within the grid delimited by [xmin, xmax] and [ymin, ymax]
	
	line2 : tuple of 2 list of 2 elements
		each value is a list of 2 elements corresponding to the x and y coordinates of an endpoint of 
line2 within the grid delimited by [xmin, xmax] and [ymin, ymax]

	xmin : float 
		minimum value on x axis

	xmax : float
		maximum value on x axis 

	ymin : float
		minimum value on y axis

	ymax : float
		maximum value on y axis

	Returns
	-------
	tuple of 2 float values
		x and y coordinates of intersection point
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


class randTruncLines():
	"""
	Class which attributes are the parameters defining 2 threshold lines.
	"""

	def __init__(self):

		nbLines = 2 # number of truncation lines set by default #TODO: add third line

		# Define endpoints of horizontal and vertical axes of truncation map
		x_endpoints = np.array([-4, 4]) 
		y_endpoints = np.array([-4, 4]) 

		lines = np.zeros((nbLines, x_endpoints.shape[0])) # array that stores the 2 endpoints of the 3 random lines to be generated

		# Define random segments each defined by a rotation angle and a distance to the origin
		plt.close()
		plt.figure()
			
		# Set rotation angle and distance to the origin for each line
#		rotationAngles = np.random.uniform(pi/2, pi/2+2*pi, nbLines) # set randomly
#		distancesToOrigin = np.random.uniform(0, 1, nbLines)
		rotationAngles = np.array([3.0087741, 7.73074376]) # set by default
		distancesToOrigin = np.array([0.55325674, 0.26806778]) 

		# For each line
		colors = ['r', 'g']
		for i in np.arange(nbLines):
			lines[i, :] = np.tan(rotationAngles[i]-pi/2)*(x_endpoints - distancesToOrigin[i]/np.cos(rotationAngles[i]))
			plt.plot(x_endpoints, lines[i, :], color=colors[i], alpha=0.7, linewidth=2)


		# Plot generated lines
		plt.xlim(x_endpoints[0], x_endpoints[1])
		plt.ylim(y_endpoints[0], y_endpoints[1])
		plt.axhline(color='k', alpha=0.8, linestyle='--', linewidth=0.2)
		plt.axvline(color='k', alpha=0.8, linestyle='--', linewidth=0.2)
		plt.savefig('truncMap.png')

		# Compute intersection between lines 
		inter_RG = segment_intersection(([x_endpoints[0], lines[0,0]], [x_endpoints[1], lines[0,1]]), 
			([x_endpoints[0], lines[1,0]],[x_endpoints[1], lines[1,1]]), x_endpoints[0], x_endpoints[1], 
			y_endpoints[0], y_endpoints[1]) # tests intersection between red and green lines

		# While all the ines don't intersect within domain, regenerate the lines
		while inter_RG == False:

			for i in np.arange(nbLines):				
				rotationAngles[i] = np.random.uniform(pi/2, pi/2+2*pi) # set randomly
				distancesToOrigin[i] = np.random.uniform(0, 1)
				lines[i, :] = np.tan(rotationAngles[i]-pi/2)*(x_endpoints - distancesToOrigin[i]/np.cos(rotationAngles[i]))
				plt.plot(x_endpoints, lines[i, :], color=colors[i], alpha=0.7, linewidth=2)
		
			# Plot randomly generated intersecting lines
			plt.xlim(x_endpoints[0], x_endpoints[1])
			plt.ylim(y_endpoints[0], y_endpoints[1])
			plt.axhline(color='k', alpha=0.8, linestyle='--', linewidth=0.2)
			plt.axvline(color='k', alpha=0.8, linestyle='--', linewidth=0.2)
			plt.savefig('truncMap.png')

			# Compute intersection point(s)
			inter_RG = segment_intersection([x_endpoints[0], lines[0,0]], [x_endpoints[1], lines[0,1]], 
				[x_endpoints[0], lines[1,0]], [x_endpoints[1], lines[1,1]], x_endpoints[0], x_endpoints[1], 
				y_endpoints[0], y_endpoints[1]) # tests intersection between red and green lines
		
		self.angles = rotationAngles
		self.dist2origin = distancesToOrigin


def thresholdLineEq(r, teta, x):
	"""
	Affine function corresponding to a threshold line defined by 2 parameters r and teta. 
	Returns the y coordinate of the point on the threshold line given an x coordinate.

	Parameters
	----------
	r : float
		distance to the origin
	
	teta : float
		rotation angle in radians 

	x : float
		x coordinate
 
	Returns
	-------
	y : float
		y coordinate 
	"""

	y = np.tan(teta-pi/2)*(x - r/np.cos(teta)) 
	
	return y
	

def truncBiGaussian24facies(g1, g2, lines):
	"""
	Truncates 2 continuous gaussian realizations into 4 facies according to 2 random threshold lines.
	Returns an array filled with values 1, 2, 3 or 4 corresponding to either of the 4 facies 
	
	Parameters
	----------
	g1 : ndarray
		numpy array filled with continuous gaussian values

	g2 : ndarray
		numpy array filled with continuous gaussian values

	lines : object 
		randTruncLines class object with attributes "angles" and "dist2origin"
		
 
	Returns
	-------
	ndarray
		numpy array of same size as g2 
	"""

	faciesGrid = g2 # initialize grid for facies simulation after truncation with values of the gaussian realization given as second argument 

	# Assign facies depending on position of pair of gaussian values on truncation map
	faciesGrid[np.where((g2 < thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & 
		(g2 > thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[0], 
		np.where((g2 < thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & 
		(g2 > thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[1]] = 1
	faciesGrid[np.where((g2 > thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & 
		(g2 > thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[0], 
		np.where((g2 > thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & 
		(g2 > thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[1]] = 2
	faciesGrid[np.where((g2 > thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & 
		(g2 < thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[0], 
		np.where((g2 > thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & 
		(g2 < thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[1]] = 3	
	faciesGrid[np.where((g2 < thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & 
		(g2 < thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[0], 
		np.where((g2 < thresholdLineEq(lines.dist2origin[0], lines.angles[0], g1)) & 
		(g2 < thresholdLineEq(lines.dist2origin[1], lines.angles[1], g1)))[1]] = 4	

	# Compute proportion of each facies
	prop_facies1 = faciesGrid.reshape(-1).tolist().count(1)/faciesGrid.size*100
	prop_facies2 = faciesGrid.reshape(-1).tolist().count(2)/faciesGrid.size*100
	prop_facies3 = faciesGrid.reshape(-1).tolist().count(3)/faciesGrid.size*100
	prop_facies4 = faciesGrid.reshape(-1).tolist().count(4)/faciesGrid.size*100
	print('Proportion of facies 1: %d%% ' % prop_facies1)
	print('Proportion of facies 2: %d%% ' % prop_facies2)
	print('Proportion of facies 3: %d%% ' % prop_facies3)
	print('Proportion of facies 4: %d%% ' % prop_facies4)

	return faciesGrid
	

def makeTruncMap(g1, g2, rule_type, thresholds):
	"""
	Plot truncation map and couples of simulated gaussian values
	"""
	
	# Define endpoints of horizontal and vertical axes of truncation map
	x_endpoints = np.array([-4, 4]) 
	y_endpoints = np.array([-4, 4]) 
	
	# Define random segments each defined by a rotation angle and a distance to the origin
	plt.close()
	plt.figure()

	plt.scatter(g1, g2, s=4, color='c', marker='o', alpha=0.3) # plot gaussian couples

	# Plot threshold lines
	if rule_type == 1:
		plt.vlines(x=thresholds[0], ymin=y_endpoints[0], ymax=y_endpoints[1],
			 color='r', linestyle='-', linewidth=2, alpha=0.7)
		plt.hlines(y=thresholds[1], xmin=thresholds[0], xmax=x_endpoints[1], 
			color='r', linestyle='-', linewidth=2, alpha=0.7)
	
	elif rule_type == 2:
		plt.vlines(x=thresholds[0], ymin=y_endpoints[0], ymax=thresholds[1], 
			color='r', linestyle='-', linewidth=2, alpha=0.7)
		plt.hlines(y=thresholds[1], xmin=x_endpoints[0], xmax=x_endpoints[1], 
			color='r', linestyle='-', linewidth=2, alpha=0.7)
	
	elif rule_type == 3:
		plt.vlines(x=thresholds[0], ymin=y_endpoints[0], ymax=y_endpoints[1], 
			color='r', linestyle='-', linewidth=2, alpha=0.7)
		plt.hlines(y=thresholds[1], xmin=x_endpoints[0], xmax=thresholds[0], 
			color='r', alpha=0.7, linestyle='-', linewidth=2)

	plt.xlim(x_endpoints[0], x_endpoints[1])
	plt.ylim(y_endpoints[0], y_endpoints[1])
	plt.axhline(color='k', alpha=0.8, linestyle='--', linewidth=0.2)
	plt.axvline(color='k', alpha=0.8, linestyle='--', linewidth=0.2)
	plt.savefig('truncMap.png')


def truncBiGaussian23facies(g1, g2, rule_type, thresholds):
	"""
	Truncates 2 continuous gaussian realizations into 3 facies according to 2 threshold values.
	Returns an array filled with values 1, 2, 3 corresponding to either of the 3 facies 
	
	Parameters
	----------
	g1 : ndarray
		numpy array filled with continuous gaussian values

	g2 : ndarray
		numpy array filled with continuous gaussian values. g1 and g2 must have the same size.

	rule_type : integer
		value is equal to either 1, 2 or 3, depending on type of truncation map

	thresholds : list of 2 floats
		first float corresponds to threshold value for g1, second float corresponds to threshold value for g2

	Returns
	-------
	ndarray
		numpy array of same size as g1 or g2 
	"""

	g1_flattened = g1.reshape(1, -1)
	g2_flattened = g2.reshape(1, -1)
	makeTruncMap(g1, g2, rule_type, thresholds) # plot truncation map and couple of gaussian values
	faciesGrid = np.zeros((g1.shape[0], g1.shape[1])) # initialize grid for facies simulation  

	# Assign facies depending on position of pair of gaussian values on truncation map
	if rule_type == 1:
		coord_facies1 = np.unravel_index(np.where(g1_flattened < thresholds[0]), (g1.shape[0], g1.shape[1]))
		coord_facies2 = np.unravel_index(np.intersect1d(np.where(g1_flattened > thresholds[0]), 
			np.where(g2_flattened > thresholds[1])), (g1.shape[0], g1.shape[1]))
		coord_facies3 = np.unravel_index(np.intersect1d(np.where(g1_flattened > thresholds[0]), 
			np.where(g2_flattened < thresholds[1])), (g1.shape[0], g1.shape[1]))

	elif rule_type == 2:
		coord_facies1 = np.unravel_index(np.where(g2_flattened > thresholds[1]), (g1.shape[0], g1.shape[1]))
		coord_facies2 = np.unravel_index(np.intersect1d(np.where(g1_flattened < thresholds[0]), 
			np.where(g2_flattened < thresholds[1])), (g1.shape[0], g1.shape[1]))
		coord_facies3 = np.unravel_index(np.intersect1d(np.where(g1_flattened > thresholds[0]), 
			np.where(g2_flattened < thresholds[1])), (g1.shape[0], g1.shape[1]))

	elif rule_type == 3:				
		coord_facies1 = np.unravel_index(np.where(g1_flattened > thresholds[0]), (g1.shape[0], g1.shape[1]))
		coord_facies2 = np.unravel_index(np.intersect1d(np.where(g1_flattened < thresholds[0]), 
			np.where(g2_flattened > thresholds[1])), (g1.shape[0], g1.shape[1]))
		coord_facies3 = np.unravel_index(np.intersect1d(np.where(g1_flattened < thresholds[0]), 
			np.where(g2_flattened < thresholds[1])), (g1.shape[0], g1.shape[1]))

	faciesGrid[coord_facies1[0], coord_facies1[1]] = 1
	faciesGrid[coord_facies2[0], coord_facies2[1]]  = 2
	faciesGrid[coord_facies3[0], coord_facies3[1]]  = 3

	# Compute proportion of each facies
	prop_facies1 = faciesGrid.reshape(-1).tolist().count(1)/faciesGrid.size*100
	prop_facies2 = faciesGrid.reshape(-1).tolist().count(2)/faciesGrid.size*100
	prop_facies3 = faciesGrid.reshape(-1).tolist().count(3)/faciesGrid.size*100
	print('Proportion of facies 1: %d%% ' % prop_facies1)
	print('Proportion of facies 2: %d%% ' % prop_facies2)
	print('Proportion of facies 3: %d%% ' % prop_facies3)

	return faciesGrid


def truncGaussian2faciesUsingThresholds(gaussianGrid, thresholds):
	"""
	Truncates the continuous gaussian realization into 2 or 3 facies according to given thresholds.
	Returns an array filled with values 1, 2 or 3 corresponding to either of the 3 facies 
	
	Parameters
	----------
	gaussianGrid : ndarray
		numpy array filled with continuous gaussian values

	thresholds : list of floats
		

	Returns
	-------
	ndarray
		numpy array of same size as gaussianGrid
	"""

	# Assign facies values according to thresholds

	faciesGrid = np.zeros((gaussianGrid.shape[0], gaussianGrid.shape[1]))

	if len(thresholds) == 1:
		faciesGrid[np.where(gaussianGrid <= thresholds[0])[0], np.where(gaussianGrid <= thresholds[0])[1]] = 1
		faciesGrid[np.where(gaussianGrid > thresholds[0])[0], np.where(gaussianGrid > thresholds[0])[1]] = 3

	# Compute proportion of each facies
	prop_facies1 = faciesGrid.reshape(-1).tolist().count(1)/faciesGrid.size*100
	prop_facies2 = faciesGrid.reshape(-1).tolist().count(2)/faciesGrid.size*100
	prop_facies3 = faciesGrid.reshape(-1).tolist().count(3)/faciesGrid.size*100
	print('Proportion of facies 1: %d%% ' % prop_facies1)
	print('Proportion of facies 2: %d%% ' % prop_facies2)
	print('Proportion of facies 3: %d%% ' % prop_facies3)

	return faciesGrid, [prop_facies1, prop_facies2, prop_facies3] 


def createSynFaciesObs(NX, NY, dx, dy, nbOfData):
	"""
	"""

	x_min = 0  # min x coordinate of data
	x_max = NX*dx # max x coordinate of data
	y_min = 0 # min y coordinate of data
	y_max = NY*dy # max y coordinate of data

	x_data = np.random.uniform(x_min, x_max, nbOfData)
	y_data = np.random.uniform(y_min, y_max, nbOfData)
	
	return x_data, y_data


def assignDataToNearestCellCentroidCoordinates(xcoord_cellCentroids, ycoord_cellCentroids, dataset):
	"""
	"""

	x_data = dataset[:, 0]
	y_data = dataset[:, 1]
	x_data2nearestCell = np.zeros(dataset.shape[0])	
	y_data2nearestCell = np.zeros(dataset.shape[0])

	i=0
	for data in zip(x_data, y_data):
		x_data2nearestCell[i] = xcoord_cellCentroids[np.abs(xcoord_cellCentroids - data[0]).argmin()]
		y_data2nearestCell[i] = ycoord_cellCentroids[np.abs(ycoord_cellCentroids - data[1]).argmin()]
		i=i+1

	return x_data2nearestCell, y_data2nearestCell


def convertFacies2IniPseudoData(faciesObs, thresholds):
	"""
	"""

	pseudoData = np.zeros(faciesObs.shape[0]) # initialize pseudo gaussian data vector

	if len(thresholds) == 1: # if two facies
		pseudoData[np.where(faciesObs == np.min(faciesObs))[0]] = -5
		pseudoData[np.where(faciesObs == np.max(faciesObs))[0]] = 5
	elif len(thresholds) == 2: # if three facies
		pseudoData[np.where(faciesObs == np.min(faciesObs))[0]] = -5
		pseudoData[np.where((faciesObs > np.min(faciesObs)) & (faciesObs < np.max(faciesObs)))[0]] = 0
		pseudoData[np.where(faciesObs == np.max(faciesObs))[0]] = 5
	
	return pseudoData


def gibbsSampling(hardDataset, thresholds, nbIter, it_st):
	"""
	"""

	# Load facies observations
	nbOfData = hardDataset.shape[0]
	faciesObs = hardDataset[:, 2]
	x_faciesObs = hardDataset[:, 0]
	y_faciesObs = hardDataset[:, 1]

	## Gibbs sampling 

	# Step 1: Initialization of the procedure

	pseudoData_ini = convertFacies2IniPseudoData(faciesObs, thresholds) # gaussian values
	np.savetxt('pseudoData_ini.txt', pseudoData_ini)

	# Step 2: Iterative procedure: Simple kriging is applied to the points in turn. For example, the value of the top point is kriged using the other three points as input data. Then, we move down to the second point and krige it using the initial values for the points below it and the new updated value for the top point. After completing the second point, we move down to the third one with is kriged using the updated values for the points above it and the old value for the point below it. When all the points have been updated by kriging, one iteration has been completed.

	pseudoData_est_allIter = np.zeros((nbOfData, nbIter))
	pseudoData_var_allIter = np.zeros((nbOfData, nbIter))
	pseudoData_residual_allIter = np.zeros((nbOfData, nbIter))
	pseudoData_allIter = np.zeros((nbOfData, nbIter+1))
	pseudoData_allIter[:, 0] = pseudoData_ini

	j=1 # initialize counter of iterations
	while j <= nbIter:
	#		print(j)
		for i in np.arange(nbOfData):
	#		print(i)
			if i == 0:
				pseudoData_est_i, pseudoData_var_i, l_i = simpleKrig(x_faciesObs[i+1:], y_faciesObs[i+1:], pseudoData_allIter[i+1:, j-1], x_faciesObs[i], y_faciesObs[i], 'spherical', 2, 0, 1)
				pseudoData_est_allIter[i, j-1] = pseudoData_est_i
				pseudoData_var_allIter[i, j-1] = pseudoData_var_i
				
			else:
				pseudoData_est_i, pseudoData_var_i, l_i = simpleKrig(np.concatenate((x_faciesObs[0:i], x_faciesObs[i+1:])), np.concatenate((y_faciesObs[0:i], y_faciesObs[i+1:])), np.concatenate((pseudoData_est_allIter[0:i, j-1], pseudoData_est_allIter[i+1:, j-1])), x_faciesObs[i], y_faciesObs[i], 'spherical', 2, 0, 1)	
				pseudoData_est_allIter[i, j-1] = pseudoData_est_i
				pseudoData_var_allIter[i, j-1] = pseudoData_var_i
		
			# Calculate the corresponding residual
			R_i = np.random.normal(0, 1)

	#		if pseudoData_ini[i] < thresholds[0]:
			if pseudoData_allIter[i, j-1] < thresholds[0]:
				while not (R_i <= -pseudoData_est_allIter[i, j-1]/pseudoData_var_allIter[i, j-1]**(1/2)):
					R_i = np.random.normal(0, 1)
			else:
				while not (R_i > -pseudoData_est_allIter[i, j-1]/pseudoData_var_allIter[i, j-1]**(1/2)):
					R_i = np.random.normal(0, 1)
			pseudoData_residual_allIter[i, j-1] = R_i

			pseudoData_allIter[i, j] = pseudoData_est_allIter[i, j-1] + pseudoData_var_allIter[i, j-1]**(1/2) * pseudoData_residual_allIter[i, j-1]

	#	print(pseudoData_allIter[:, j-1])
		j=j+1

	pseudoData_final = pseudoData_allIter[:, it_st]
	np.savetxt('pseudoData_iter' + str(it_st) + '.txt', pseudoData_final, fmt='%.2f')
	#	print('\n')
	#	print(pseudoData_est_allIter)
	#	print('\n')
	#	print(pseudoData_var_allIter)

	fig, axarray = plt.subplots(5, 2, figsize=(9, 9))
	k=0
	for i in np.arange(5):
		for j in np.arange(2):
			axarray[i, j].scatter(np.arange(nbIter+1), pseudoData_allIter[k, :])
			k=k+1

	plt.show()

	return pseudoData_final


def findDataCellCoordinates(x_data, y_data, XX, YY):
	"""
	"""
	
	nbOfData = x_data.shape[0]
	cellData_lin_index = np.zeros(nbOfData)
	cellData_col_index = np.zeros(nbOfData)

	for k in np.arange(nbOfData): # get cell coordinates of data
		cellData_lin_index[k] = np.intersect1d(np.where(XX == x_data[k])[0], np.where(YY == y_data[k])[0])
		cellData_col_index[k] = np.intersect1d(np.where(XX == x_data[k])[1], np.where(YY == y_data[k])[1])

	return cellData_lin_index.astype(int), cellData_col_index.astype(int)
