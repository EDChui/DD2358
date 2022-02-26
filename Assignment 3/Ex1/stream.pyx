import numpy as np
cimport numpy as np

def copy(double[:] a, double[:] b, double[:] c, int size):
	cdef unsigned int i
	for i in range(size):
		c[i] = a[i]

def scale(double[:] a, double[:] b, double[:] c, int size, double scalar):
	cdef unsigned int i
	for i in range(size):
		b[i] = scalar * c[i]

def sum(double[:] a, double[:] b, double[:] c, int size):
	cdef unsigned int i
	for i in range(size):
		c[i] = a[i] + b[i]

def triad(double[:] a, double[:] b, double[:] c, int size, double scalar):
	cdef unsigned int i
	for i in range(size):
		a[i] = b[i] + scalar * c[i]
