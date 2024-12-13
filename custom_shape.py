# For extra credit, create your own wireframe and include an image of
# in the written part of your solution.  For full credit, you must
# make something sufficiently complex, and you must give it a name.

straight_square_edges = [
    [(-1, 2, -2), (1, 2, -2)],
    [(2, 1, -2), (2, -1, -2)],
    [(-1, -2, -2), (1, -2, -2)],
    [(-2, 1, -2), (-2, -1, -2)],
    [(-1, 2, 2), (1, 2, 2)],
    [(2, 1, 2), (2, -1, 2)],
    [(-1, -2, 2), (1, -2, 2)],
    [(-2, 1, 2), (-2, -1, 2)],
]

verticals = [
    [(1, -2, -2), (-1, -2, 2)],
    [(-1, -2, -2), (-2, -1, 2)],
    [(-2, -1, -2), (-2, 1, 2)],
    [(-2, 1, -2), (-1, 2, 2)],
    [(-1, 2, -2), (1, 2, 2)],
    [(1, 2, -2), (2, 1, 2)],
    [(2, 1, -2), (2, -1, 2)],
    [(2, -1, -2), (1, -2, 2)],
]

octagon_diagonal_edges = [
    [(1, 2, -2), (2, 1, -2)],
    [(2, -1, -2), (1, -2, -2)],
    [(-1, -2, -2), (-2, -1, -2)],
    [(-2, 1, -2), (-1, 2, -2)],
    [(1, 2, 2), (2, 1, 2)],
    [(2, -1, 2), (1, -2, 2)],
    [(-1, -2, 2), (-2, -1, 2)],
    [(-2, 1, 2), (-1, 2, 2)],
]

extra_1 = [
    [(-1, 2, -2.2), (1, 2, -2.2)],
    [(2, 1, -2.2), (2, -1, -2.2)],
    [(-1, -2, -2.2), (1, -2, -2.2)],
    [(-2, 1, -2.2), (-2, -1, -2.2)],
    [(-1, 2, 2.2), (1, 2, 2.2)],
    [(2, 1, 2.2), (2, -1, 2.2)],
    [(-1, -2, 2.2), (1, -2, 2.2)],
    [(-2, 1, 2.2), (-2, -1, 2.2)],
]

extra_2 = [
    [(1, 2, -2.2), (2, 1, -2.2)],
    [(2, -1, -2.2), (1, -2, -2.2)],
    [(-1, -2, -2.2), (-2, -1, -2.2)],
    [(-2, 1, -2.2), (-1, 2, -2.2)],
    [(1, 2, 2.2), (2, 1, 2.2)],
    [(2, -1, 2.2), (1, -2, 2.2)],
    [(-1, -2, 2.2), (-2, -1, 2.2)],
    [(-2, 1, 2.2), (-1, 2, 2.2)],
]

extra_3_vertical = [
    [(-1, 2, -2), (-1, 2, -2.2)],
    [(1, 2, -2), (1, 2, -2.2)],
    [(2, 1, -2), (2, 1, -2.2)],
    [(2, -1, -2), (2, -1, -2.2)],
    [(-1, -2, -2), (-1, -2, -2.2)],
    [(1, -2, -2), (1, -2, -2.2)],
    [(-2, 1, -2), (-2, 1, -2.2)],
    [(-2, -1, -2), (-2, -1, -2.2)],
    [(-1, 2, 2), (-1, 2, 2.2)],
    [(1, 2, 2), (1, 2, 2.2)],
    [(2, 1, 2), (2, 1, 2.2)],
    [(2, -1, 2), (2, -1, 2.2)],
    [(-1, -2, 2), (-1, -2, 2.2)],
    [(1, -2, 2), (1, -2, 2.2)],
    [(-2, 1, 2), (-2, 1, 2.2)],
    [(-2, -1, 2), (-2, -1, 2.2)],
]

custom_shape = straight_square_edges + octagon_diagonal_edges + verticals + extra_1 + extra_2 + extra_3_vertical