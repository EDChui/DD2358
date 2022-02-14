import pytest
from Ex3_JuliaSet import calc_pure_python

def get_test_calc_pure_python_data():
	return [(1000, 300, 33219980)]

@pytest.mark.parametrize("desired_width, max_iterations, expected", get_test_calc_pure_python_data())
def test_calc_pure_python(desired_width, max_iterations, expected):
	output = calc_pure_python(desired_width, max_iterations)
	assert sum(output) == expected
