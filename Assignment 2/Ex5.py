import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from tqdm import tqdm

EXP_RANGE = range(8, 1025)
NUM_OF_EXP = 10

def create_N_by_N_DFT_matrix(N: int):
	# A reference is made when creating this method.
	# Ref: https://stackoverflow.com/a/19739589
	i, j = np.meshgrid(np.arange(N), np.arange(N))
	omega = np.exp(-2 * np.pi * 1j / N)
	W = np.power(omega, i*j) / np.sqrt(N)
	return W

def create_random_input_signal(N: int):
	return np.random.random(N) + np.random.random(N) *1j

def DFT(x):
	"""This DFT uses the matrix-vector multiplication X = Wx to find the DFT of the input signal x.

	Args:
		x: Input signal, a vector.

	Return:
		X: The DFT of the signal.
	"""
	N = len(x)
	W = create_N_by_N_DFT_matrix(N)
	X = np.matmul(W, x)
	return X

def DFT_optimized(x):
	"""This DFT uses the the second equations given in the homework assignment to find the DFT of the input signal x.

	Args:
		x: Input singal, a vector.

	Returns:
		X: The DFT of the signal.
	"""
	N = len(x)
	X = np.zeros(N, dtype=np.cdouble)
	
	omega = np.exp(-2 * np.pi * 1j/ N)
	for k in range(N):
		y = np.power(np.power(omega, k), np.arange(N))
		X[k] = np.matmul(x, y) / np.sqrt(N)

	return X

def DFT_numpy(x):
	return np.fft.fft(x, norm="ortho")

def run_experiment(dftFunc):
	avgTimeSpents = []
	for N in tqdm(EXP_RANGE, desc=f"Running {dftFunc.__name__}", ncols=100):
		timeSpents = []

		for i in range(NUM_OF_EXP):
			x = create_random_input_signal(N)
			startTime = timer()
			dftFunc(x)
			timeSpents.append(timer() - startTime)

		avgTimeSpents.append(np.average(timeSpents))

	return avgTimeSpents


if __name__ == "__main__":
	timeSpents = run_experiment(DFT)
	plt.plot(EXP_RANGE, timeSpents, label="DFT")
	
	timeSpents = run_experiment(DFT_optimized)
	plt.plot(EXP_RANGE, timeSpents, label="DFT Optimized")

	# timeSpents = run_experiment(DFT_numpy)
	# plt.plot(EXP_RANGE, timeSpents, label="DFT NumPy")

	plt.title("DFT Execution time")
	plt.xlabel("Input size")
	plt.ylabel("Average time spent (s)")
	plt.legend()
	plt.show()
