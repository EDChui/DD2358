import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cythonfn

GRID_SIZES = (5, 10, 25, 50, 100, 125, 250, 500)
SEIDEL_ITERATIONS = 1000

def create_random_grid(size: int):
	grid = np.random.rand(size, size).astype(np.double)
	# Boundaries are set to zero.
	grid[[0, -1]] = 0
	grid[:, [0, -1]] = 0
	return grid

def run_GS_solver(grid, size: int):
	startTime = timer()
	for i in range(SEIDEL_ITERATIONS):
		grid = cythonfn.gauss_seidel(grid)
	return timer() - startTime

if __name__ == "__main__":
	timeSpents = []
	for gridSize in tqdm(GRID_SIZES):
		grid = create_random_grid(gridSize)
		timeSpents.append(run_GS_solver(grid, gridSize))
	
	plt.plot(GRID_SIZES, timeSpents, label="Cython")
	plt.title("The performance of the Gauss-Seidel solver")
	plt.xlabel("Grid Sizes")
	plt.ylabel("Time spent (s)")
	plt.legend()
	plt.show()
