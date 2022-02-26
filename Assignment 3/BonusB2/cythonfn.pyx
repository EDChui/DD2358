import numpy as np
cimport numpy as np

def update(frameNum, grid, N):
	cdef int ON = 255
	cdef int OFF = 0
	# A reference is made when creating this method.
	# Ref: https://www.labri.fr/perso/nrougier/from-python-to-numpy/#the-game-of-life

	# Counting the number of neighbours.
	neighbourCount = np.zeros(grid.shape)
	neighbourCount[1:-1, 1:-1] += ( grid[ :-2, :-2] + grid[ :-2, 1:-1] + grid[ :-2, 2: ] + 
									grid[1:-1, :-2] +                    grid[1:-1, 2: ] + 
									grid[2:  , :-2] + grid[2:  , 1:-1] + grid[2:  , 2: ]) / ON

	# Apply rules.
	birth = (neighbourCount==3)[1:-1,1:-1] & (grid[1:-1,1:-1]==OFF)
	survive = ((neighbourCount==2) | (neighbourCount==3))[1:-1,1:-1] & (grid[1:-1,1:-1]==ON)
	grid[...] = OFF
	grid[1:-1,1:-1][birth | survive] = ON
