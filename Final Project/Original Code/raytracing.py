import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from functools import wraps
from math import ceil
from tqdm import tqdm

# Sphere properties.
position = np.array([0., 0., 1.])
radius = 1.
color = np.array([0., 0., 1.])
diffuse = 1.
specular_c = 1.
specular_k = 50

# Light position and color.
L = np.array([5., 5., -10.])
color_light = np.ones(3)
ambient = .05

# Camera.
O = np.array([0., 0., -1.])  # Position.
Q = np.array([0., 0., 0.])   # Pointing to.

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

def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection
    # of the ray (O, D) with the sphere (S, R), or
    # +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a
    # normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 \
            else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def trace_ray(O, D):
    # Find first point of intersection with the scene.
    t = intersect_sphere(O, D, position, radius)
    # No intersection?
    if t == np.inf:
        return
    # Find the point of intersection on the object.
    M = O + D * t
    N = normalize(M - position)
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Ambient light.
    col = ambient
    # Lambert shading (diffuse).
    col += diffuse * max(np.dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col += specular_c * color_light * \
        max(np.dot(N, normalize(toL + toO)), 0) \
        ** specular_k
    return col

def run(w, h):
    img = np.zeros((h, w, 3))
    # Loop through all pixels.
    for i, x in enumerate(np.linspace(-1, 1, w)):
        for j, y in enumerate(np.linspace(-1, 1, h)):
            # Position of the pixel.
            Q[0], Q[1] = x, y
            # Direction of the ray going through
            # the optical center.
            D = normalize(Q - O)
            # Launch the ray and get the color
            # of the pixel.
            col = trace_ray(O, D)
            if col is None:
                continue
            img[h - j - 1, i, :] = np.clip(col, 0, 1)
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
    mins = []
    maxs = []

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
    plt.plot(sizes, averages, label=f"Original", color="blue")
    plt.title("Ray Tracing")
    plt.xlabel("Input size")
    plt.ylabel("Average time spent (s)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # get_image(w=400, h=400)
    test_performance()
