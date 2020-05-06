import numpy as np
from timeit import timeit


def generate_points_py(w, h, dtype=np.int):
    points = []
    for x in range(w):
        for y in range(h):
            points.append([x, y])

    return np.array(points, dtype=dtype)


def generate_points_np(w, h, dtype=np.int):
    xs, ys = np.mgrid[0: w, 0: h]
    # points = np.stack((xs, ys), axis=2).reshape(-1, 2).astype(dtype)
    points = np.stack((xs.ravel(), ys.ravel()), axis=1).astype(dtype)       # To avoid reshape
    return points


if __name__ == '__main__':
    w, h = 10, 8
    points_1 = generate_points_py(w, h)
    points_2 = generate_points_np(w, h)
    print("Array equal:", np.array_equal(points_1, points_2))

    w, h = 1500, 1500
    print("testing for large array w:{}, h:{}:".format(w, h))
    py_time = timeit("generate_points_py(w, h)", number=1, globals=globals())
    np_time = timeit("generate_points_np(w, h)", number=1, globals=globals())
    print("using python:", py_time )
    print("using numpy:", np_time)
    print("faster {:.2f} times".format(py_time / np_time))
