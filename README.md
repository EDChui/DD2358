# DD2358

---

# Assignment 1

- [Bonus - CPU Profiler](./Assignment%201/cpuProfiler.py)

	- [README](./Assignment%201/README.md)

# Assignment 2

- [Ex 1 - Check HPC libraries used by NumPy installation](./Assignment%202/Ex1.py)

- [Ex 2 - STREAM Benchmark in Python to Measure the Memory Bandwidth](./Assignment%202/Ex2.py)

- [Ex 3 - PyTest with the Julia Set Code](./Assignment%202/Ex3_JuliaSet.py)

	- [Test](./Assignment%202/test_Ex3_JuliaSet.py)

- [Ex 4 - Python DGEMM Benchmark Operation](./Assignment%202/Ex4.py)

	- [Test](./Assignment%202/test_Ex4.py)

- [Ex 5 - Python Discrete Fourier Transform](./Assignment%202/Ex5.py)

	- [Test](./Assignment%202/test_Ex5.py)

- [Bonus - Performance Analysis and Optimization of the Game of Life Code](./Assignment%202/Bonus_conway.py)

# Assignment 3

- [Ex 1 - Cythonize the STREAM Benchmark](./Assignment%203/Ex1)

- [Ex 2 - Gauss-Seidel for Poisson Solver](./Assignment%203/Ex2)

- [Bonus 1 - Optimize Ex 2 in C (with cffi)](./Assignment%203/Ex2/Bonus)

	- Use `gcc -fPIC -shared -o libgs.so gauss_seidel.c` to compile the C code into `.so` file.

- [Bonus 2 - Cythonize the Game of Life](./Assignment%203/BonusB2)

- [Bonus 3 - Optimize Game of Life in C (with cffi)](./Assignment%203/BonusB3)

	- Use `gcc -fPIC -shared -o libupdate.so update.c` to compile the C code into `.so` file.

# Final Project - Ray Tracing Engine

- [Original Code](./Final%20Project/Original%20Code)

- [Optimized Version (with cffi)](./Final%20Project/Optimized)

	- Use `gcc -fPIC -shared -o libraytracing.so c_raytracing.c` to compile the C code into `.so` file.

	- Use `python -m test_c_raytracing.py` to run the pytest unit test.
