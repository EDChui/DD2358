import pytest
import numpy as np
from cffi import FFI
import raytracing as rt

# CFFI
ffi = FFI()
ffi.cdef("""void vector_elementwise_add(double* result, double* a, double* b, int size);
void vector_elementwise_subtract(double* result, double* a, double* b, int size);
void vector_elementwise_multiply(double* result, double* a, double* b, int size);
void vector_elementwise_scale(double* result, double* a, double scale, int size);
void vector_normalize(double* result, double* x, int size);
double vector_dot_product(double* a, double* b, int size);
double min(double a, double b);
double max(double a, double b);
double clip(double a, double min, double max);
double intersect_sphere(double* O, double* D, double* S, double R);
int trace_ray(double* col, double* O, double* D, double* L, double* position, double* color, double* color_light, double radius, double ambient, double diffuse, double specular_c, double specular_k);
void run(double* img, int w, int h, double* O, double* L, double* position, double* color, double* color_light, double radius, double ambient, double diffuse, double specular_c, double specular_k);""")
lib = ffi.dlopen("./libraytracing.so")

# Sphere properties.
position = np.array([0., 0., 1.])
radius = 1.
color = np.array([0., 0., 1.])
diffuse = 1.
specular_c = 1.
specular_k = 50

c_position = ffi.cast("double*", position.ctypes.data)
c_radius = ffi.cast("double", radius)
c_color = ffi.cast("double*", color.ctypes.data)
c_diffuse = ffi.cast("double", diffuse)
c_specular_c = ffi.cast("double", specular_c)
c_specular_k = ffi.cast("double", specular_k)

# Light position and color.
L = np.array([5., 5., -10.])
color_light = np.ones(3)
ambient = .05

c_L = ffi.cast("double*", L.ctypes.data)
c_color_light = ffi.cast("double*", color_light.ctypes.data)
c_ambient = ffi.cast("double", ambient)

# Camera.
O = np.array([0., 0., -1.])  # Position.
Q = np.array([0., 0., 0.])   # Pointing to.

c_O = ffi.cast("double*", O.ctypes.data)
c_Q = ffi.cast("double*", Q.ctypes.data)

def get_test_vector_elementwise_add_data():
	return [
		([1., 2., 3.], [1., 1., 1.], [2., 3., 4.]),
		([1., 1., 1.], [.1, .23, -1.5], [1.1, 1.23, -0.5]),
		([-1.1, -2.2, -3.], [10., 0., 15.2], [8.9, -2.2, 12.2]),
		([1., 2., 3., 4.], [1., 1., 1., 1.], [2., 3., 4., 5.]),
		]

@pytest.mark.parametrize("a, b, expected", get_test_vector_elementwise_add_data())
def test_vector_elementwise_add(a, b, expected):
	a = np.array(a)
	b = np.array(b)
	c = np.zeros(a.size)
	expected = np.array(expected)

	c_a = ffi.cast("double*", a.ctypes.data)
	c_b = ffi.cast("double*", b.ctypes.data)
	c_c = ffi.cast("double*", c.ctypes.data)

	lib.vector_elementwise_add(c_c, c_a, c_b, a.size)
	np.testing.assert_allclose(c, expected)

def get_test_vector_elementwise_subtract_data():
	return [
		([1., 2., 3.], [1., 1., 1.], [0., 1., 2.]),
		([1., 1., 1.], [.1, .23, -1.5], [0.9, 0.77, 2.5]),
		([-1.1, -2.2, -3.], [10., 0., 15.2], [-11.1, -2.2, -18.2]),
		([1., 2., 3., 4.], [1., 1., 1., 1.], [0., 1., 2., 3.]),
		]

@pytest.mark.parametrize("a, b, expected", get_test_vector_elementwise_subtract_data())
def test_vector_elementwise_subtract(a, b, expected):
	a = np.array(a)
	b = np.array(b)
	c = np.zeros(a.size)
	expected = np.array(expected)

	c_a = ffi.cast("double*", a.ctypes.data)
	c_b = ffi.cast("double*", b.ctypes.data)
	c_c = ffi.cast("double*", c.ctypes.data)

	lib.vector_elementwise_subtract(c_c, c_a, c_b, a.size)
	np.testing.assert_allclose(c, expected)

def get_test_vector_elementwise_multiply_data():
	return [
		([1., 2., 3.], [1., 1., 1.], [1., 2., 3.]),
		([1., 1., 1.], [.1, .23, -1.5], [.1, .23, -1.5]),
		([-1.1, -2.2, -3.], [10., 0., 15.2], [-11., 0., -45.6]),
		([1., 2., 3., 4.], [1., 1., 1., 1.], [1., 2., 3., 4.]),
		]

@pytest.mark.parametrize("a, b, expected", get_test_vector_elementwise_multiply_data())
def test_vector_elementwise_multiply(a, b, expected):
	a = np.array(a)
	b = np.array(b)
	c = np.zeros(a.size)
	expected = np.array(expected)

	c_a = ffi.cast("double*", a.ctypes.data)
	c_b = ffi.cast("double*", b.ctypes.data)
	c_c = ffi.cast("double*", c.ctypes.data)

	lib.vector_elementwise_multiply(c_c, c_a, c_b, a.size)
	np.testing.assert_allclose(c, expected)

def get_test_vector_elementwise_scale_data():
	return [
		([1., 2., 3.], 1, [1., 2., 3.]),
		([1., 2., 3.], -1, [-1., -2., -3.]),
		([-1.1, -2.2, -3.], 1.5, [-1.65, -3.3, -4.5]),
		([1., 2., 3., 4.], 1, [1., 2., 3., 4.]),
		]

@pytest.mark.parametrize("a, scale, expected", get_test_vector_elementwise_scale_data())
def test_vector_elementwise_scale(a, scale, expected):
	a = np.array(a)
	result = np.zeros(a.size)
	expected = np.array(expected)

	c_a = ffi.cast("double*", a.ctypes.data)
	c_scale = ffi.cast("double", scale)
	c_result = ffi.cast("double*", result.ctypes.data)

	lib.vector_elementwise_scale(c_result, c_a, c_scale, a.size)
	np.testing.assert_allclose(result, expected)


def get_test_vector_normalize_data():
	return [
		([1., 2., 3.], [0.26726124, 0.53452248, 0.80178373]),
		([4., 5., -6.], [0.45584231, 0.56980288, -0.68376346]),
		([7.6, 22., -6.2], [0.31551843,  0.91334282, -0.25739661]),
		]

@pytest.mark.parametrize("a, expected", get_test_vector_normalize_data())
def test_vector_normalize(a, expected):
	a = np.array(a)
	result = np.zeros(a.size)
	expected = np.array(expected)

	c_a = ffi.cast("double*", a.ctypes.data)
	c_result = ffi.cast("double*", result.ctypes.data)
	lib.vector_normalize(c_result, c_a, a.size)
	np.testing.assert_allclose(result, expected)

def get_test_vector_dot_product_data():
	return [
		([1., 2., 3.], [4., 5., 6.], 32),
		([1.6, 2.5, 3.5], [4.2, 5.2, 6.7], 43.17),
		([-1.6, 2.5, -3.5], [-4.2, 5.8, -6.5], 43.97)
		]

@pytest.mark.parametrize("a, b, expected", get_test_vector_dot_product_data())
def test_vector_dot_product(a, b, expected):
	a = np.array(a)
	b = np.array(b)

	c_a = ffi.cast("double*", a.ctypes.data)
	c_b = ffi.cast("double*", b.ctypes.data)
	result = lib.vector_dot_product(c_a, c_b, a.size)
	np.testing.assert_allclose(result, expected)

def get_test_min_data():
	return [
		(1, 2, 1),
		(0.5, 0.3, 0.3),
		(-12, 12, -12),
		(-1.5, 4.5, -1.5),
		]

@pytest.mark.parametrize("a, b, expected", get_test_min_data())
def test_min(a, b, expected):
	c_a = ffi.cast("double", a)
	c_b = ffi.cast("double", b)
	result = lib.min(c_a, c_b)
	np.testing.assert_allclose(result, expected)

def get_test_max_data():
	return [
		(1, 2, 2),
		(0.5, 0.3, 0.5),
		(-12, 12, 12),
		(-1.5, 4.5, 4.5),
		]

@pytest.mark.parametrize("a, b, expected", get_test_max_data())
def test_max(a, b, expected):
	c_a = ffi.cast("double", a)
	c_b = ffi.cast("double", b)
	result = lib.max(c_a, c_b)
	np.testing.assert_allclose(result, expected)

def get_test_clip_data():
	return [
		(0.3, 0, 1, 0.3), 
		(-1.2, 0, 1, 0), 
		(3, 0, 1, 1),
		(0.2123, -5, 10, 0.2123),
	]

@pytest.mark.parametrize("a, min, max, expected", get_test_clip_data())
def test_clip(a, min, max, expected):
	c_a = ffi.cast("double", a)
	c_min = ffi.cast("double", min)
	c_max = ffi.cast("double", max)
	result = lib.clip(c_a, c_min, c_max)
	np.testing.assert_allclose(result, expected)

def get_test_intersect_sphere_data():
	return [
		([1, 1, 1], np.inf),
		([0, 0, -0.5], np.inf),
		([0, 0, 0.5], 2.0),
		([0, 0, 0.23], 4.3478260869565215),
	]

@pytest.mark.parametrize("D, expected", get_test_intersect_sphere_data())
def test_intersect_sphere(D, expected):
	D = np.array(D)
	c_D = ffi.cast("double*", D.ctypes.data)
	result = lib.intersect_sphere(c_O, c_D, c_position, c_radius)

	# As there is no inf in C, so assume if the distance is > 1e200 is inf.
	if (result > 1e200):
		result = np.inf
	np.testing.assert_allclose(result, expected)

def get_test_trace_ray_data():
	return [
		([1., 1., 1.], None),
		([0., 0., -0.5], None),
		([0., 0., 0.5], [0.14018093, 0.14018093, 0.95667751]),
		([0., 0., 0.23], [0.14018093, 0.14018093, 0.95667751]),
	]

@pytest.mark.parametrize("D, expected", get_test_trace_ray_data())
def test_trace_ray(D, expected):
	D = np.array(D)
	result = np.zeros(3)
	c_D = ffi.cast("double*", D.ctypes.data)
	c_result = ffi.cast("double*", result.ctypes.data)

	success = lib.trace_ray(c_result, c_O, c_D, c_L, c_position, c_color, c_color_light, c_radius, c_ambient, c_diffuse, c_specular_c, c_specular_k)
	
	if (success == 0):
		result = None
		assert result == expected
	else:
		np.testing.assert_allclose(result, expected)

def get_test_run_data():
	return [
		(3,
		[
			[
				[0.        , 0.        , 0.        ],
				[0.        , 0.        , 0.        ],
				[0.        , 0.        , 0.        ]
				],
			[
				[0.        , 0.        , 0.        ],
				[0.14018093, 0.14018093, 0.95667751],
				[0.        , 0.        , 0.        ]
				],
			[
				[0.        , 0.        , 0.        ],
				[0.        , 0.        , 0.        ],
				[0.        , 0.        , 0.        ]
			]
		]),
		(4,
		[
			[
				[0.        , 0.        , 0.        ],
				[0.        , 0.        , 0.        ],
				[0.        , 0.        , 0.        ],
				[0.        , 0.        , 0.        ]
				],
			[
				[0.        , 0.        , 0.        ],
				[0.05      , 0.05      , 0.71130764],
				[0.05094038, 0.05094038, 1.        ],
				[0.        , 0.        , 0.        ]
				],
			[
				[0.        , 0.        , 0.        ],
				[0.05      , 0.05      , 0.38977643],
				[0.05      , 0.05      , 0.71130764],
				[0.        , 0.        , 0.        ]
				],
			[	[0.        , 0.        , 0.        ],
				[0.        , 0.        , 0.        ],
				[0.        , 0.        , 0.        ],
				[0.        , 0.        , 0.        ]
				]
			]
		)
	]

@pytest.mark.parametrize("size, expected", get_test_run_data())
def test_run(size, expected):
	img = np.zeros(size * size * 3, dtype=np.double)
	expected = np.array(expected)
	c_img = ffi.cast("double*", img.ctypes.data)
	c_size = ffi.cast("int", size)
	
	lib.run(c_img, c_size, c_size, c_O, c_L, c_position, c_color, c_color_light, c_radius, c_ambient, c_diffuse, c_specular_c, c_specular_k)
	
	img = img.reshape((size, size, 3))
	np.testing.assert_allclose(img, expected)