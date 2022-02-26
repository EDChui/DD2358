from unittest import result
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from cffi import FFI

GRID_SIZES = (5, 10, 25, 50, 100, 125, 250, 500)
SEIDEL_ITERATION = 1000

ffi = FFI()
ffi.cdef("void gauss_seidel(double** grid, int size, int numIteration);")
lib = ffi.dlopen("./libgs.so")

def create_random_grid(size: int):
	grid = np.random.rand(size, size).astype(np.double)
	# Boundaries are set to zero.
	grid[[0, -1]] = 0
	grid[:, [0, -1]] = 0
	return grid

def gauss_seidel(cGrid, cSize, cNumIteration):
	lib.gauss_seidel(cGrid, cSize, cNumIteration)
	return cGrid

def run_GS_solver(grid, size: int):
	startTime = timer()

	# Covert to C type object.
	cGrid = ffi.new("double* [%d]" % (size))
	for i in range(size):
		cGrid[i] = ffi.cast("double *", grid[i].ctypes.data)
	
	cSize = ffi.cast("int", size)
	cNumIteration = ffi.cast("int", SEIDEL_ITERATION)
	
	cResult = gauss_seidel(cGrid, cSize, cNumIteration)

	# Covert back to NumPy array.
	result = grid.copy()
	for i in range(size):
		result[i] = np.frombuffer(ffi.buffer(cResult[i], size*result.dtype.itemsize), dtype=np.double)
	return timer() - startTime

if __name__ == "__main__":
	timeSpents = []
	for gridSize in tqdm(GRID_SIZES):
		grid = create_random_grid(gridSize)
		timeSpents.append(run_GS_solver(grid, gridSize))
	
	plt.plot(GRID_SIZES, timeSpents, label="CIFF")
	plt.title("The performance of the Gauss-Seidel solver")
	plt.xlabel("Grid Sizes")
	plt.ylabel("Time spent (s)")
	plt.legend()
	plt.show()
