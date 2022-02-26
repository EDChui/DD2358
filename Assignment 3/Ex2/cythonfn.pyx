import numpy as np
cimport numpy as np

def gauss_seidel(double[:, :] grid):
	cdef double[:, :] newGrid = grid.copy()
	cdef unsigned int i, j
	for i in range(1, grid.shape[0] - 1):
		for j in range(1, grid.shape[1] - 1):
			newGrid[i, j] = 0.25 * (grid[i+1, j] + newGrid[i-1, j] + grid[i, j+1] + newGrid[i, j-1])	
	return newGrid
