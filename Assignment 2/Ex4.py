from typing import Tuple, List
from array import array
import numpy as np 
from numexpr import evaluate
from timeit import default_timer as timer
from tqdm import tqdm
import matplotlib.pyplot as plt

MATRIX_SIZES = (1, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 250, 375, 500)
# MATRIX_SIZES = (1, 5, 10, 25, 50, 100, 200, 375, 500, 1000, 5000, 10_000)			# For comparing with and without numexpr.
NUM_OF_EXP = 10

def init_list_matrices(size: int) -> Tuple[List[list]]:
	a = [np.random.rand(size) for i in range(size)] 
	b = [np.random.rand(size) for i in range(size)] 
	c = [np.random.rand(size) for i in range(size)] 
	return a, b, c

def init_array_matrices(size: int):
	# It is not possible to create a 2D array using the array module.
	# Thus, a 1D array with size size*size is made instead.
	# See: https://stackoverflow.com/questions/62531869/multidimensional-array-using-module-array-in-python
	a = array("d", np.random.rand(size*size))
	b = array("d", np.random.rand(size*size))
	c = array("d", np.random.rand(size*size))
	return a, b, c

def init_numpy_matrices(size: int):
	a = np.random.rand(size, size).astype(np.double)
	b = np.random.rand(size, size).astype(np.double)
	c = np.random.rand(size, size).astype(np.double)
	return a, b, c

def dgemm_list(a, b, c, size):
	for i in range(size):
		for j in range(size):
			for k in range(size):
				c[i][j] += a[i][k] * b[k][j]
	return c

def dgemm_array(a, b, c, size):
	for i in range(size):
		for j in range(size):
			for k in range(size):
				c[i*size+j] += a[i*size+k] * b[k*size+j]
	return c

def dgemm_numpy(a, b, c, size):
	c += np.matmul(a, b)
	return c

def dgemm_numexpr(a, b, c, size):
	# Note that numexpr can only be used for element-wise operation.
	# See: https://groups.google.com/g/numexpr/c/jDc6bRSvh6c
	temp = np.matmul(a, b)
	evaluate("temp + c", out=c)
	return c

def run_experiment(matrixSize, initMatrixFunc, dgemmFunc):
	a, b, c = initMatrixFunc(matrixSize)
	startTime = timer()
	dgemmFunc(a, b, c, matrixSize)
	timeSpent = timer() - startTime
	return timeSpent


if __name__ == "__main__":
	avgTimeSpents = []
	stdTimeSpents = []
	minTimeSpents = []
	maxTimeSpents = []

	for matrixSize in MATRIX_SIZES:
		timeSpents = []

		for t in tqdm(range(NUM_OF_EXP), desc=f"Running\t{matrixSize}*{matrixSize}\tmatrix", ncols=100):
			# timeSpent = run_experiment(matrixSize, init_list_matrices, dgemm_list)
			# timeSpent = run_experiment(matrixSize, init_array_matrices, dgemm_array)
			# timeSpent = run_experiment(matrixSize, init_numpy_matrices, dgemm_numpy)
			timeSpent = run_experiment(matrixSize, init_numpy_matrices, dgemm_numexpr)

			timeSpents.append(timeSpent)
		
		avgTimeSpents.append(np.average(timeSpents))
		stdTimeSpents.append(np.std(timeSpents))
		minTimeSpents.append(min(timeSpents))
		maxTimeSpents.append(max(timeSpents))

	# Output result
	print("Size\t"+"\t\t".join([f"{i}*{i}" for i in MATRIX_SIZES]))	
	print("Avg\t"+"\t".join([f"{i:.4E}" for i in avgTimeSpents]))
	print("Std\t"+"\t".join([f"{i:.4E}" for i in stdTimeSpents]))
	print("Min\t"+"\t".join([f"{i:.4E}" for i in minTimeSpents]))
	print("Max\t"+"\t".join([f"{i:.4E}" for i in maxTimeSpents]))

	# Plot
	# plt.plot(MATRIX_SIZES, avgTimeSpents, label="Python Lists")
	# plt.plot(MATRIX_SIZES, avgTimeSpents, label="Array Module")
	# plt.plot(MATRIX_SIZES, avgTimeSpents, label="NumPy Array")
	plt.plot(MATRIX_SIZES, avgTimeSpents, label="NumPy Array w/ numexpr")

	plt.title("Execution time")
	plt.xlabel("Matrix size N*N")
	plt.ylabel("Average time spent (s)")
	plt.legend()
	plt.show()
