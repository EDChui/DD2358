import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from functools import wraps
from math import ceil
from tqdm import tqdm
from cffi import FFI

# CFFI
ffi = FFI()
ffi.cdef("""void run(double* img, int w, int h, double* O, double* L, double* position, double* color, double* color_light, double radius, double ambient, double diffuse, double specular_c, double specular_k);""")
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

def timerFn(fn):
    # Decorator for measuring the time spent on a function.
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timer()
        result = fn(*args, **kwargs)
        t2 = timer()
        print(f"@timerFun: {fn.__name__} took {t2 - t1} seconds")
        return result
    return measure_time

def normalize(x):
    # This function normalizes a vector.
    x /= np.linalg.norm(x)
    return x

def run(w, h):
    img = np.zeros(h * w * 3, dtype=np.double)

    # Convert to C type objects.
    c_img = ffi.cast("double*", img.ctypes.data)
    c_w = ffi.cast("int", w)
    c_h = ffi.cast("int", h)

    lib.run(c_img, c_w, c_h, c_O, c_L, c_position, c_color, c_color_light, c_radius, c_ambient, c_diffuse, c_specular_c, c_specular_k)

    img = img.reshape((h, w, 3))
    return img

def get_image(w, h):
    img = run(w, h)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    ax.set_axis_off()
    plt.show()

def test_performance():
    num_test_per_size = 10
    sizes = [10, 25, 50, 75, 100, 125, 150, 250, 375, 500, 750, 875, 1000, 1250, 1375, 1500]
    averages = []
    stds = []
    mins =[]
    maxs =[]

    # Get the time spent on run with varying size.
    for size in sizes:
        times = []
        for i in tqdm(range(num_test_per_size), desc=f"Testing {size}*{size}...", ncols=100):
            startTime = timer()
            run(w=size, h=size)
            times.append(timer() - startTime)
        averages.append(np.average(times))
        stds.append(np.std(times))
        mins.append(np.min(times))
        maxs.append(np.max(times))

    # Output a table about the performance.
    for i in range(ceil(len(sizes) / 5)):
        print("Size\t"+"\t\t".join([f"{size}" for size in sizes[i*5: (i+1)*5+1]]))
        print("Avg\t"+"\t".join([f"{i:.4E}" for i in averages[i*5: (i+1)*5+1]]))
        print("Std\t"+"\t".join([f"{i:.4E}" for i in stds[i*5: (i+1)*5+1]]))
        print("Min\t"+"\t".join([f"{i:.4E}" for i in mins[i*5: (i+1)*5+1]]))
        print("Max\t"+"\t".join([f"{i:.4E}" for i in maxs[i*5: (i+1)*5+1]]))
        print()

    # Plot the average.
    plt.plot(sizes, averages, label=f"Optimized", color="orange")
    plt.title("Ray Tracing")
    plt.xlabel("Input size")
    plt.ylabel("Average time spent (s)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # get_image(w=400, h=400)
    test_performance()
