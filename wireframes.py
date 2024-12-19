import numpy as np

###############
# WIRE FRAMES #
###############

cube = [[(-1, -1, -1), (1, -1, -1)],
        [(1, -1, -1), (1, 1, -1)],
        [(1, 1, -1), (-1, 1, -1)],
        [(-1, 1, -1), (-1, -1, -1)],
        [(-1, -1, -1), (-1, -1, 1)],
        [(1, -1, -1), (1, -1, 1)],
        [(1, 1, -1), (1, 1, 1)],
        [(-1, 1, -1), (-1, 1, 1)],
        [(-1, -1, 1), (1, -1, 1)],
        [(1, -1, 1), (1, 1, 1)],
        [(1, 1, 1), (-1, 1, 1)],
        [(-1, 1, 1), (-1, -1, 1)]]

pyramid = [[(-1, -1, -1), (-1, -1, 1)],
           [(-1, -1, 1), (1, -1, 1)],
           [(1, -1, 1), (1, -1, -1)],
           [(1, -1, -1), (-1, -1, -1)],
           [(-1, -1, -1), (0, 1, 0)],
           [(-1, -1, 1), (0, 1, 0)],
           [(1, -1, 1), (0, 1, 0)],
           [(1, -1, -1), (0, 1, 0)]]

def mk_circle(r, n, z):
    out = []
    for i in range(n):
        out.append(((r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n), z),
                    (r * np.cos(2 * np.pi * (i + 1) / n), r * np.sin(2 * np.pi * (i + 1) / n), z)))
    return out

def mk_wheel_edge(r, n, z1, z2):
    out = []
    for i in range(n):
        out.append(((r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n), z1),
                    (r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n), z2)))
    return out

def mk_wheel_face(r1, r2, n, z):
    out = []
    for i in range(n):
        out.append(((r1 * np.cos(2 * np.pi * i / n), r1 * np.sin(2 * np.pi * i / n), z),
                    (r2 * np.cos(2 * np.pi * i / n), r2 * np.sin(2 * np.pi * i / n), z)))
    return out

wheel = mk_circle(3, 50, 1) + \
    mk_circle(1, 50, 1) + \
    mk_circle(3, 50, -1) + \
    mk_circle(1, 50, -1) + \
    mk_wheel_edge(3, 50, 1, -1) + \
    mk_wheel_edge(1, 50, 1, -1) + \
    mk_wheel_face(1, 3, 50, 1) + \
    mk_wheel_face(1, 3, 50, -1)