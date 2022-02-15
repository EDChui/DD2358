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

def init_list_matrices(size: int):
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
	
	# Another design that does not use np.matmul, but the performance is really poor :( .
	#
	# bTranspose = np.transpose(b)
	# for i in range(size):
	# 	for j in range(size):
	# 		x = a[i]
	# 		y = bTranspose[j]
	# 		c[i, j] += evaluate("sum(x * y)")

	temp = np.matmul(a, b)
	evaluate("temp + c", out=c)
	return c

def run_experiment(initMatrixFunc, dgemmFunc) -> Tuple[list]:
	avgTimeSpents = []
	stdTimeSpents = []
	minTimeSpents = []
	maxTimeSpents = []

	for matrixSize in MATRIX_SIZES:
		timeSpents = []
		for t in tqdm(range(NUM_OF_EXP), desc=f"Running {dgemmFunc.__name__} - {matrixSize}*{matrixSize} matrix", ncols=100):
			a, b, c = initMatrixFunc(matrixSize)
			startTime = timer()
			dgemmFunc(a, b, c, matrixSize)
			timeSpents.append(timer() - startTime)

		avgTimeSpents.append(np.average(timeSpents))
		stdTimeSpents.append(np.std(timeSpents))
		minTimeSpents.append(min(timeSpents))
		maxTimeSpents.append(max(timeSpents))
	
	return avgTimeSpents, stdTimeSpents, minTimeSpents, maxTimeSpents

def output_stats(stats):
	print("Size\t"+"\t\t".join([f"{i}*{i}" for i in MATRIX_SIZES]))	
	print("Avg\t"+"\t".join([f"{i:.4E}" for i in stats[0]]))
	print("Std\t"+"\t".join([f"{i:.4E}" for i in stats[1]]))
	print("Min\t"+"\t".join([f"{i:.4E}" for i in stats[2]]))
	print("Max\t"+"\t".join([f"{i:.4E}" for i in stats[3]]))

if __name__ == "__main__":
	expStats = run_experiment(init_list_matrices, dgemm_list)
	output_stats(expStats)
	plt.plot(MATRIX_SIZES, expStats[0], label="Python Lists")
	
	expStats = run_experiment(init_array_matrices, dgemm_array)
	output_stats(expStats)
	plt.plot(MATRIX_SIZES, expStats[0], label="Array Module")
	
	expStats = run_experiment(init_numpy_matrices, dgemm_numpy)
	output_stats(expStats)
	plt.plot(MATRIX_SIZES, expStats[0], label="NumPy Array")
	
	expStats = run_experiment(init_numpy_matrices, dgemm_numexpr)
	output_stats(expStats)
	plt.plot(MATRIX_SIZES, expStats[0], label="NumPy Array w/ numexpr")

	plt.title("Execution time")
	plt.xlabel("Matrix size N*N")
	plt.ylabel("Average time spent (s)")
	plt.legend()
	plt.show()
