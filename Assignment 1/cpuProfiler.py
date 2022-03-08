from numpy import average
import psutil
import threading
import matplotlib.pyplot as plt
from functools import wraps

class CPUProfiler:
	interval = 1				# Default interval is 1 second.
	cpuPercentages = []			# A List containing a list of CPU percentages taken in every interval.
	__isRunning = False
	__currentThread = None

	def  __init__(self, interval:int=1) -> None:
		self.interval = interval

	# Return the running state of the profiler.
	def isRunning(self) -> bool:
		return self.__isRunning

	# Method that keeps recoding the CPU usage percentage every interval. Runs in another therad.
	def __run(self) -> None:
		while self.__isRunning:
			self.cpuPercentages.append(psutil.cpu_percent(interval=self.interval, percpu=True))

	# Start the CPU profiling.
	def start(self) -> None:
		self.__isRunning = True
		if self.__currentThread is None:
			self.__currentThread = threading.Thread(target=self.__run)
			self.__currentThread.start()

	# Stop the CPU profiling.
	def stop(self) -> None:
		self.__isRunning = False
		if self.__currentThread is not None:
			self.__currentThread.join()
			self.__currentThread = None

	# Clear all CPU usage percentage records.
	def clear(self) -> None:
		self.cpuPercentages = []

	# Plot the evolution of the CPU percentage.
	def plot(self) -> None:
		for i in range(psutil.cpu_count()):
			plt.plot([percentages[i] for percentages in self.cpuPercentages], label="CPU"+str(i))
		plt.title("CPU percentage, sampling in every "+str(self.interval)+" second(s)")
		plt.legend()
		plt.show()

	# Generate a simple summary table.
	def generateTable(self) -> None:
		print("Number of core(s): "+str(psutil.cpu_count()))
		print("Interval of sampling: "+str(self.interval)+" second(s).")
		print("Total number of intervals taken: "+str(len(self.cpuPercentages))+".")
		print("Name\tMin\tMax\tAverage")
		print("="*50)
		for i in range(psutil.cpu_count()):
			cpuStats = [percentages[i] for percentages in self.cpuPercentages]
			if len(cpuStats) == 0:
				continue
			print("CPU"+str(i), end="\t")
			print("{:.2f}".format(min(cpuStats)), end="\t")
			print("{:.2f}".format(max(cpuStats)), end="\t")
			print("{:.2f}".format(average(cpuStats)))

# Example of Function Decorator.
def cpuProfilerFn(fn):
	@wraps(fn)
	def runCPUProfiler(*args, **kwargs):
		result = None
		cpuProfiler = CPUProfiler(interval=1)
		cpuProfiler.start()
		try:
			result = fn(*args, **kwargs)
		except KeyboardInterrupt:
			cpuProfiler.stop()
		finally:
			cpuProfiler.stop()
			cpuProfiler.plot()
			cpuProfiler.generateTable()
			return result
	return runCPUProfiler
