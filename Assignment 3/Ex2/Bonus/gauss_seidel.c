#include <stdio.h>
#include <stdlib.h>

void gauss_seidel(double** grid, int size, int numIteration) {
	// Create a new grid for calculation.
	double** newGrid = (double**)malloc(size* sizeof(double*));
	for (int i=0; i<size; i++) {
		newGrid[i] = (double*)malloc(size * sizeof(double));
	}

	double** temp = NULL;

	// Pointers for clean up.
	double** result = grid;
	double** originNewGrid = newGrid;

	// Actual calculation.
	for (int t=0; t<numIteration; t++) {
		for (int i=1; i<size-1; i++) {
			for (int j=1; j<size-1; j++) {
				newGrid[i][j] = 0.25 * (grid[i+1][j] + newGrid[i-1][j] + grid[i][j+1] + newGrid[i][j-1]);
			}
		}
		temp = newGrid;
		newGrid = grid;
		grid = temp;
	}

	// Make sure the correct result is returned.
	if (result != grid) {
		for (int i=0; i<size; i++) {
			for (int j=0; j<size; j++) {
				result[i][j] = grid[i][j];
			}
		}
	}

	// Free memory of the new grid.
	for (int i=0; i<size; i++) {
		free(originNewGrid[i]);
	}
	free(originNewGrid);
}
