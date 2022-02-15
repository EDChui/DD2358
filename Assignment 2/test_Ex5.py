import pytest
import numpy as np
from Ex5 import DFT, DFT_optimized

def get_test_data():
	return [
		(np.array([1, 2, 3], dtype=np.cdouble), np.array([3.46410162+0j, -0.8660254+0.5j, -0.8660254-0.5j], dtype=np.cdouble)),
		(np.array([1, 1, 1], dtype=np.cdouble), np.array([1.73205081+0j, 0+0j, 0+0j], dtype=np.cdouble)),
		(np.array([23+1j, -12-92j, 12+0.3j], dtype=np.cdouble), np.array([13.27905619-52.36566942j, -32.87094381+39.04886011j, 59.42905619+15.04886011j], dtype=np.cdouble))
	]

@pytest.mark.parametrize("x, expected", get_test_data())
def test_DFT(x, expected):
	result = DFT(x)
	np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize("x, expected", get_test_data())
def test_DFT_optimized(x, expected):
	result = DFT_optimized(x)
	np.testing.assert_array_almost_equal(result, expected)
