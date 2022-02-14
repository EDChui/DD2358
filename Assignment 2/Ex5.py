import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from tqdm import tqdm

EXP_RANGE = range(1024, 1025)
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
	"""This DFT uses the matrix-vector multiplication X = Wx to find the DFT of the signal x.

	Args:
		x: Input signal, a vector.

	Return:
		X: the DFT of the signal.
	"""
	N = len(x)
	W = create_N_by_N_DFT_matrix(N)
	X = np.matmul(W, x)
	return X

def run_experiment(dftFunc):
	timeSpentsList = []
	for N in tqdm(EXP_RANGE, desc=f"Running {dftFunc.__name__}", ncols=100):
		timeSpents = []

		for i in range(NUM_OF_EXP):
			x = create_random_input_signal(N)
			startTime = timer()
			dftFunc(x)
			timeSpents.append(timer() - startTime)

		timeSpentsList.append(np.average(timeSpents))

	return timeSpentsList


if __name__ == "__main__":
	# timeSpents = run_experiment(DFT)
	# plt.plot(EXP_RANGE, timeSpents, label="DFT")
	# plt.title("DFT Execution time")
	# plt.xlabel("Input size")
	# plt.ylabel("Average time spent (s)")
	# plt.legend()
	# plt.show()


	x = create_random_input_signal(1024)
	X = DFT(x)
