import pytest
from array import array
import numpy as np
from Ex4 import dgemm_list, dgemm_array, dgemm_numpy, dgemm_numexpr


def get_test_matrix_mult_data():
	return [([[1.0]*3 for i in range(3)], [[2.0]*3 for i in range(3)], [[3.0]*3 for i in range(3)], 3, [[9.0]*3 for i in range(3)]),
			([[1.0, 7.0], [2.0, 4.0]], [[3.0, 3.0], [5.0, 2.0]], [[1.0, -1.0], [-1.0, 1.0]], 2, [[39.0, 16.0], [25.0, 15.0]])
			]


@pytest.mark.parametrize("a, b, c, size, expected", get_test_matrix_mult_data())
def test_dgemm_list(a, b, c, size, expected):
	result = dgemm_list(a, b, c, size)
	np.testing.assert_allclose(result, expected)

@pytest.mark.parametrize("a, b, c, size, expected", get_test_matrix_mult_data())
def test_dgemm_array(a, b, c, size, expected):
	a = array("d", [item for sublist in a for item in sublist])
	b = array("d", [item for sublist in b for item in sublist])
	c = array("d", [item for sublist in c for item in sublist])
	expected = array("d", [item for sublist in expected for item in sublist])

	result = dgemm_array(a, b, c, size)
	for i in range(size):
		for j in range(size):
			np.testing.assert_allclose(result[i*size+j], expected[i*size+j])

@pytest.mark.parametrize("a, b, c, size, expected", get_test_matrix_mult_data())
def test_dgemm_numpy(a, b, c, size, expected):
	a = np.array(a)
	b = np.array(b)
	c = np.array(c)
	expected = np.array(expected)
	
	result = dgemm_numpy(a, b, c, size)
	np.testing.assert_allclose(result, expected)

@pytest.mark.parametrize("a, b, c, size, expected", get_test_matrix_mult_data())
def test_dgemm_numexpr(a, b, c, size, expected):
	a = np.array(a)
	b = np.array(b)
	c = np.array(c)
	expected = np.array(expected)
	
	result = dgemm_numexpr(a, b, c, size)
	np.testing.assert_allclose(result, expected)
