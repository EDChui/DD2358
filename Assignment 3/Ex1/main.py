import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import sys
import stream

ARRAY_SIZES = (1, 5, 10, 50, 100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000)
INIT_A_VAL = 1.0
INIT_B_VAL = 2.0
INIT_C_VAL = 0.0
SCALAR = 2.0

def init_numpy_arrays(size: int):
	a = np.full(size, INIT_A_VAL, dtype=np.double)
	b = np.full(size, INIT_B_VAL, dtype=np.double)
	c = np.full(size, INIT_C_VAL, dtype=np.double)
	return a, b, c

def get_function_exec_time(function, *args, **kwargs):
	startTime = timer()
	function(*args, **kwargs)
	return timer() - startTime

def get_operations_exec_time(a, b, c, size):
	times = []
	times.append(get_function_exec_time(stream.copy, a, b, c, size))
	times.append(get_function_exec_time(stream.scale, a, b, c, size, SCALAR))
	times.append(get_function_exec_time(stream.sum, a, b, c, size))
	times.append(get_function_exec_time(stream.triad, a, b, c, size, SCALAR))
	return times

def calc_memory_bandwidth(arrayType, arraySize, times):
	memoryBandwidths = []

	# Copy
	memoryBandwidths.append((2 * sys.getsizeof(arrayType) * arraySize / 2**20) / times[0])	
	# Add
	memoryBandwidths.append((2 * sys.getsizeof(arrayType) * arraySize / 2**20) / times[1])
	# Scale
	memoryBandwidths.append((3 * sys.getsizeof(arrayType) * arraySize / 2**20) / times[2])
	# Triad
	memoryBandwidths.append((3 * sys.getsizeof(arrayType) * arraySize / 2**20) / times[3])

	return memoryBandwidths

if __name__ == "__main__":
	performancesList = []

	for arraySize in ARRAY_SIZES:
		a, b, c = init_numpy_arrays(arraySize)
		executionTimes = get_operations_exec_time(a, b, c, arraySize)
		performancesList.append(calc_memory_bandwidth(type(a), arraySize, executionTimes))

	for idx, operation in enumerate(("copy", "add", "scale", "triad")):
		plt.plot(ARRAY_SIZES, [performances[idx] for performances in performancesList], label="Python Lists - "+operation)
	
	plt.title("STREAM Benchmark")
	plt.xlabel("Stream Array Sizes")
	plt.ylabel("Memory Bandwidth (MB/seconds)")
	plt.legend()
	plt.show()