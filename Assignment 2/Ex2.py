from typing import Tuple, List
from array import array
import numpy as np
from timeit import default_timer as timer
import sys
import matplotlib.pyplot as plt

STREAM_ARRAY_SIZES = (1, 5, 10, 50, 100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000)
INIT_A_VAL = 1.0
INIT_B_VAL = 2.0
INIT_C_VAL = 0.0
SCALAR = 2.0

def init_python_lists(size: int) -> Tuple[list]:
	a = [INIT_A_VAL] * size
	b = [INIT_B_VAL] * size
	c = [INIT_C_VAL] * size
	return a, b, c

def init_array_module_arrays(size: int) -> Tuple[array]:
	pla, plb, plc = init_python_lists(size)
	a = array('d', pla)
	b = array('d', plb)
	c = array('d', plc)
	return a, b, c

def init_numpy_arrays(size: int) -> Tuple[np.ndarray]:
	a = np.full(size, INIT_A_VAL, dtype=np.double)
	b = np.full(size, INIT_B_VAL, dtype=np.double)
	c = np.full(size, INIT_C_VAL, dtype=np.double)
	return a, b, c

def time_the_operations(a, b, c, size: int) -> List[float]:
	times = []

	# Copy
	startTime = timer()
	for j in range(size):
		c[j] = a[j]
	times.append(timer() - startTime)

	# Scale
	startTime = timer()
	for j in range(size):
		b[j] = SCALAR * c[j]
	times.append(timer() - startTime)
	
	# Sum
	startTime = timer()
	for j in range(size):
		c[j] = a[j] + b[j]
	times.append(timer() - startTime)

	# Triad
	startTime = timer()
	for j in range(size):
		a[j] = b[j] + SCALAR * c[j]
	times.append(timer() - startTime)
	return times

def calc_memory_bandwidth(arrayType, arraySize: int, times: Tuple[float]) -> List[float]:
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

def stream_benchmark(size: int, initArrFunc) -> List[float]:
	a, b, c = initArrFunc(size)
	times = time_the_operations(a, b, c, size)
	memoryBandwidths = calc_memory_bandwidth(type(a), size, times)
	return memoryBandwidths


if __name__ == "__main__":
	performancesList = []

	for streamArraySize in STREAM_ARRAY_SIZES:
		performancesList.append(stream_benchmark(streamArraySize, init_python_lists))
		# performancesList.append(stream_benchmark(streamArraySize, init_array_module_arrays))
		# performancesList.append(stream_benchmark(streamArraySize, init_numpy_arrays))

	# Plot
	for idx, operation in enumerate(("copy", "add", "scale", "triad")):
		plt.plot(STREAM_ARRAY_SIZES, [performances[idx] for performances in performancesList], label="Python Lists - "+operation)
		# plt.plot(STREAM_ARRAY_SIZES, [performances[idx] for performances in performancesList], label="Array Module Arrays - "+operation)
		# plt.plot(STREAM_ARRAY_SIZES, [performances[idx] for performances in performancesList], label="NumPy Arrays - "+operation)

	plt.title("STREAM Benchmark")
	plt.xlabel("Stream Array Sizes")
	plt.ylabel("Memory Bandwidth (MB/seconds)")
	plt.legend()
	plt.show()
