#include <stdio.h>
#include <stdlib.h>

const int ON = 255;
const int OFF = 0;

void update(int** grid, int** newGrid, int size) {
	int total = 0;

	for (int i=0; i<size; i++) {
		for (int j=0; j<size; j++) {
			newGrid[i][j] = grid[i][j];
			total = 0;
			for (int h=-1; h<2; h++) {
				for (int k=-1; k<2; k++) {
					if (!((i+h) < 0 || (i+h) > size-1 || (j+k) < 0 || (j+k) > size-1 || (h==0 && k ==0))) {
						total += grid[i+h][j+k];
					}
				}
			}
			total = (int) (total / ON);
			
			if (grid[i][j] == ON) {
				if (total < 2 || total > 3) {
					newGrid[i][j] = OFF;
				}
			} else {
				if (total == 3) {
					newGrid[i][j] = ON;
				}
			}
		}
	}

	for (int i=0; i<size; i++) {
		for (int j=0; j<size; j++) {
			grid[i][j] = newGrid[i][j];
		}
	}
}